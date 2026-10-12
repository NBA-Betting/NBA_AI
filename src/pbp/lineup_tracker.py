"""
Lineup tracker: determines which 5 players are on court at each moment.

Handles:
- Period starter detection from PBP events (no API calls)
- Substitution tracking for both older and v3 formats
- Player name → ID resolution for older format subs
"""

from __future__ import annotations

import logging
from collections import defaultdict

from .event_parser import _SUB_RE, Event

logger = logging.getLogger(__name__)


class LineupTracker:
    """
    Tracks which players are on court for each team throughout a game.

    Usage:
        tracker = LineupTracker(events, home_team_id, away_team_id)
        tracker.initialize()
        for event in events:
            lineup = tracker.get_current_players()  # {team_id: [p1, p2, ...]}
            tracker.process_event(event)
    """

    def __init__(
        self,
        events: list[Event],
        home_team_id: int,
        away_team_id: int,
        player_names: dict[int, tuple[int, str]] | None = None,
    ):
        """
        Args:
            events: parsed PBP events for one game
            home_team_id: home team numeric ID
            away_team_id: away team numeric ID
            player_names: optional {player_id: (team_id, last_name)} from PlayerBox.
                          Seeds the name map to resolve subs for players with no
                          non-substitution events (bench warmers).
        """
        self.events = events
        # Ensure team IDs are int (DB stores TEXT, PBP parser returns int)
        self.home_team_id = int(home_team_id)
        self.away_team_id = int(away_team_id)
        self.team_ids = {self.home_team_id, self.away_team_id}

        # Current on-court players: {team_id: set of player_ids}
        self.on_court: dict[int, set[int]] = {
            self.home_team_id: set(),
            self.away_team_id: set(),
        }

        # Build name → player_id lookup from all events
        self._name_to_id: dict[tuple[int, str], int] = {}  # (team_id, name) → pid
        self._name_to_id_global: dict[str, int] = {}  # name → pid (no team)

        # Game roster: {team_id: set of player_ids} from PlayerBox (all who played)
        self._game_roster: dict[int, set[int]] = {
            self.home_team_id: set(),
            self.away_team_id: set(),
        }
        # Track all players ever seen on court (for roster inference)
        self._ever_on_court: dict[int, set[int]] = {
            self.home_team_id: set(),
            self.away_team_id: set(),
        }

        # Seed from external player names (e.g., PlayerBox) first
        # player_names values can be (team_id, name) or (team_id, [name1, name2, ...])
        if player_names:
            for pid, (tid, names) in player_names.items():
                pid = int(pid)
                tid = int(tid)
                name_list = names if isinstance(names, list) else [names]
                for name in name_list:
                    name_lower = name.lower()
                    self._name_to_id[(tid, name_lower)] = pid
                    self._name_to_id_global[name_lower] = pid
                if tid in self._game_roster:
                    self._game_roster[tid].add(pid)

        self._build_name_lookup()

        # Track whether starters were successfully found for each period
        self.starters_found: dict[int, bool] = {}
        self._period_subbed_in: set[int] = set()

        # Group events by period
        self._period_events: dict[int, list[Event]] = defaultdict(list)
        for evt in events:
            self._period_events[evt.period].append(evt)

    def _build_name_lookup(self):
        """Build player name → ID mapping from all events in the game."""
        for evt in self.events:
            if evt.player_id and evt.player_id != 0 and evt.team_id:
                # Use playerNameI or playerName from raw data via description
                name = self._extract_player_name(evt)
                if name:
                    key = (evt.team_id, name.lower())
                    self._name_to_id[key] = evt.player_id
                    self._name_to_id_global[name.lower()] = evt.player_id

            # Also capture assist, block, steal player names if available
            if evt.assist_player_id:
                pass  # Already have the ID
            if evt.steal_player_id:
                pass

    def _extract_player_name(self, evt: Event) -> str:
        """Extract the player's last name from the event."""
        # The description often has the last name at the start
        # e.g., "Ball 11' Driving Floating Jump Shot" or "MISS Ball 11'..."
        desc = evt.description
        if not desc:
            return ""
        # Older-format substitution "SUB: In FOR Out": the row's player is the
        # outgoing one, so its name is the "Out" part. This teaches the lookup
        # players whose only actions are substitutions (else their entry can't
        # be resolved and a later sub-out makes them a phantom period starter)
        if evt.is_substitution:
            m = _SUB_RE.search(desc)
            return m.group(2).strip() if m else ""
        # First word that looks like a name (skip MISS, SUB, etc.)
        parts = desc.split()
        for part in parts:
            if part.upper() not in (
                "MISS",
                "SUB:",
                "FOR",
                "REBOUND",
                "BLOCK",
                "STEAL",
                "TURNOVER",
                "FOUL",
                "TIMEOUT",
                "VIOLATION",
            ):
                return part
        return ""

    def detect_period_starters(self, period: int) -> dict[int, set[int]]:
        """
        Detect which 5 players started a period by walking PBP events.

        Algorithm (from pbpstats start_of_period.py):
        1. Walk all events in the period
        2. Players appearing as player_id (not via sub, tech, ejection) are starters
        3. Also capture assist/block/steal/foul-drawn player IDs as starters
        4. Players entering via substitution are NOT starters
        5. Need exactly 5 per team
        """
        period_events = self._period_events.get(period, [])
        if not period_events:
            self.starters_found[period] = False
            return {tid: set() for tid in self.team_ids}

        starters: list[int] = []
        subbed_in: set[int] = set()
        player_team: dict[int, int] = {}

        for evt in period_events:
            pid = evt.player_id
            tid = evt.team_id

            # Jump-ball players are on court (teams assigned from their other
            # events, else filled in below)
            for jb_pid in evt.jump_ball_player_ids:
                if jb_pid not in starters and jb_pid not in subbed_in:
                    starters.append(jb_pid)

            if pid == 0 or not pid:
                continue
            if tid not in self.team_ids and tid != 0:
                continue

            # Track player→team mapping
            if tid in self.team_ids and not evt.is_substitution:
                player_team[pid] = tid

            # Substitution: mark incoming as subbed-in, outgoing as potential starter
            if evt.is_substitution:
                if evt.sub_direction == "in" and evt.sub_in_player_id:
                    in_pid = evt.sub_in_player_id
                    if in_pid not in starters:
                        subbed_in.add(in_pid)
                    if tid in self.team_ids:
                        player_team[in_pid] = tid
                elif evt.sub_direction == "out" and evt.sub_out_player_id:
                    out_pid = evt.sub_out_player_id
                    if out_pid not in starters and out_pid not in subbed_in:
                        starters.append(out_pid)
                    if tid in self.team_ids:
                        player_team[out_pid] = tid
                elif evt.sub_direction == "both":
                    # Older format: personId = outgoing
                    out_pid = evt.sub_out_player_id
                    if out_pid and out_pid not in starters and out_pid not in subbed_in:
                        starters.append(out_pid)
                    if tid in self.team_ids and out_pid:
                        player_team[out_pid] = tid
                    # Resolve incoming player from name
                    in_pid = self._resolve_sub_incoming(evt)
                    if in_pid:
                        if in_pid not in starters:
                            subbed_in.add(in_pid)
                        if tid in self.team_ids:
                            player_team[in_pid] = tid
                continue

            # Skip techs, ejections, FTs at period start
            is_tech = evt.is_foul and evt.is_technical
            is_ejection = evt.action_type == "ejection"
            is_period_start_ft = (
                evt.is_free_throw
                and evt.clock_seconds >= 719.0  # within 1 sec of 12:00
            )
            if is_tech or is_ejection or is_period_start_ft:
                continue

            # Add player as starter if not already known
            if pid not in starters and pid not in subbed_in:
                starters.append(pid)

            # Also capture secondary players (assist, block, steal, foul drawn)
            for secondary_pid in [
                evt.assist_player_id,
                evt.block_player_id,
                evt.steal_player_id,
                evt.foul_drawn_player_id,
            ]:
                if (
                    secondary_pid
                    and secondary_pid not in starters
                    and secondary_pid not in subbed_in
                ):
                    starters.append(secondary_pid)
                    # Infer team for secondary players
                    if secondary_pid not in player_team:
                        # Assist/foul-drawn = same team as primary
                        # Block/steal = opposite team
                        if secondary_pid == evt.assist_player_id:
                            if tid in self.team_ids:
                                player_team[secondary_pid] = tid
                        elif secondary_pid in (
                            evt.block_player_id,
                            evt.steal_player_id,
                        ):
                            opp = self._opposite_team(tid)
                            if opp:
                                player_team[secondary_pid] = opp
                        elif secondary_pid == evt.foul_drawn_player_id:
                            opp = self._opposite_team(tid)
                            if opp:
                                player_team[secondary_pid] = opp

        self._period_subbed_in = subbed_in

        # Split starters by team
        starters_by_team: dict[int, set[int]] = {tid: set() for tid in self.team_ids}
        unassigned = []

        for pid in starters:
            tid = player_team.get(pid)
            if tid in self.team_ids:
                starters_by_team[tid].add(pid)
            else:
                unassigned.append(pid)

        # Try to assign unassigned players to teams that need them
        for pid in unassigned:
            for tid in self.team_ids:
                if len(starters_by_team[tid]) < 5:
                    starters_by_team[tid].add(pid)
                    break

        # Trim to 5 per team (take first 5 if we have more)
        for tid in self.team_ids:
            players = starters_by_team[tid]
            if len(players) > 5:
                starters_by_team[tid] = set(list(players)[:5])

        # Validate
        success = all(len(starters_by_team[tid]) == 5 for tid in self.team_ids)
        self.starters_found[period] = success

        if not success:
            counts = {tid: len(starters_by_team[tid]) for tid in self.team_ids}
            logger.debug(f"Period {period}: incomplete starters {counts}")

        return starters_by_team

    def initialize_period(self, period: int):
        """Set the on-court players for the start of a period."""
        previous = {tid: set(players) for tid, players in self.on_court.items()}
        starters = self.detect_period_starters(period)
        # A player with no action at all in the period (e.g. an OT where he
        # never touches the ball) can't be detected: he stayed on from the
        # end of the previous period, unless he subbed in during this one
        if not self.starters_found[period]:
            taken = set().union(*starters.values())
            for tid, players in starters.items():
                for pid in sorted(previous.get(tid, ())):
                    if len(players) >= 5:
                        break
                    if pid not in taken and pid not in self._period_subbed_in:
                        players.add(pid)
            self.starters_found[period] = all(
                len(starters[tid]) == 5 for tid in self.team_ids
            )
        self.on_court = {tid: set(players) for tid, players in starters.items()}
        # Track starters as seen on court
        for tid, players in self.on_court.items():
            self._ever_on_court[tid].update(players)

    def process_substitution(self, evt: Event):
        """Update on-court players for a substitution event."""
        tid = evt.team_id
        if tid not in self.team_ids:
            return

        if evt.sub_direction == "out":
            self.on_court[tid].discard(evt.sub_out_player_id)
        elif evt.sub_direction == "in":
            self.on_court[tid].add(evt.sub_in_player_id)
            self._ever_on_court[tid].add(evt.sub_in_player_id)
        elif evt.sub_direction == "both":
            # Older format: outgoing player ID is known, incoming resolved by name
            if evt.sub_out_player_id:
                self.on_court[tid].discard(evt.sub_out_player_id)
            in_pid = self._resolve_sub_incoming(evt)
            if in_pid:
                self.on_court[tid].add(in_pid)
                self._ever_on_court[tid].add(in_pid)
            elif len(self.on_court[tid]) < 5 and self._game_roster.get(tid):
                # Roster inference: find who from the game roster hasn't been
                # on court yet — they must be the incoming player
                candidates = self._game_roster[tid] - self._ever_on_court[tid]
                if len(candidates) == 1:
                    inferred = candidates.pop()
                    self.on_court[tid].add(inferred)
                    self._ever_on_court[tid].add(inferred)
                    logger.debug(
                        f"Inferred incoming sub: player {inferred} "
                        f"(team {tid}, only unseen roster player)"
                    )

    def get_current_players(self) -> dict[int, set[int]]:
        """Get current on-court players: {team_id: set of player_ids}."""
        return self.on_court

    def get_all_on_court(self) -> set[int]:
        """Get all 10 on-court player IDs."""
        result = set()
        for players in self.on_court.values():
            result.update(players)
        return result

    def _resolve_sub_incoming(self, evt: Event) -> int:
        """Resolve incoming player ID for older format substitutions."""
        incoming_name = evt.descriptor  # stored by from_older()
        if not incoming_name:
            return 0

        tid = evt.team_id
        name_lower = incoming_name.lower()

        # Try team-specific lookup first
        pid = self._name_to_id.get((tid, name_lower), 0)
        if pid:
            return pid

        # Try global lookup
        pid = self._name_to_id_global.get(name_lower, 0)
        if pid:
            return pid

        # Try partial match: stored name is substring of incoming or vice versa
        # (handles "hardaway" ↔ "hardaway jr.", "mark morris" ↔ "markieff morris")
        for (t, name), p in self._name_to_id.items():
            if t == tid and (name in name_lower or name_lower in name):
                return p

        # Try first-word match: incoming "hardaway jr." → first word "hardaway"
        first_word = name_lower.split()[0] if name_lower else ""
        if first_word and len(first_word) > 1:
            pid = self._name_to_id.get((tid, first_word), 0)
            if pid:
                return pid
            # Global first-word fallback
            pid = self._name_to_id_global.get(first_word, 0)
            if pid:
                return pid

        # Try last-word match: incoming "Mark Morris" → "morris"
        words = name_lower.split()
        if len(words) > 1:
            last_word = words[-1]
            pid = self._name_to_id.get((tid, last_word), 0)
            if pid:
                return pid

        # Prefix match: incoming "ha" matches stored "ha seung-jin"
        # (handles truncated PBP names)
        if len(name_lower) >= 2:
            for (t, name), p in self._name_to_id.items():
                if t == tid and name.startswith(name_lower):
                    return p

        logger.debug(
            f"Could not resolve incoming sub player: '{incoming_name}' " f"(team {tid})"
        )
        return 0

    def _opposite_team(self, team_id: int) -> int:
        """Get the opposing team ID."""
        for tid in self.team_ids:
            if tid != team_id:
                return tid
        return 0

    @property
    def all_starters_found(self) -> bool:
        """True if all periods had successful starter detection."""
        return all(self.starters_found.values())
