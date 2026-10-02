# -*- coding: utf-8 -*-
"""Carregamento e validação de dados de partidas a partir do banco SQLite."""

import sqlite3
import pandas as pd


def load_matches_df(db_path: str = "futebol.db") -> pd.DataFrame:
    """Carrega todas as partidas ordenadas estritamente por data e match_id.
    
    A ordenação `date ASC, match_id ASC` garante a integridade causal na passada
    única cronológica.
    """
    conn = sqlite3.connect(db_path)
    query = """
        SELECT match_id, date, season, competition_id, round,
               home_team_id, away_team_id, home_goals, away_goals, result,
               home_rest_days, away_rest_days,
               home_odds, draw_odds, away_odds, odds_open
        FROM matches
        ORDER BY date ASC, match_id ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Garantia de tipos
    df["match_id"] = df["match_id"].astype(int)
    df["date"] = df["date"].astype(str)
    df["competition_id"] = df["competition_id"].astype(int)
    df["home_team_id"] = df["home_team_id"].astype(int)
    df["away_team_id"] = df["away_team_id"].astype(int)
    
    return df
