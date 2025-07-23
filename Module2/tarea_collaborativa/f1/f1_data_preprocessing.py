import pandas as pd
from api_f1_call import get_super_times_by_season


def build_super_time_dataframe(races: pd.DataFrame, super_times: dict, by: str = "driver"):
    """
    Crea un DataFrame completo per visualizzazione scatter-line.
    
    Args:
        season: stagione selezionata
        by: "driver" o "team"

    Returns:
        pd.DataFrame con colonne standardizzate:
            race_index, race_name, entity_name, image_url, time, time_ms, superTimeRatio, superTimeDelta%
    """
    assert by in ["driver", "team"], "Valore 'by' deve essere 'driver' o 'team'"

    
    all_rows = []
    
    # Ottieni i dati delle gare per associare i nomi
    race_name_map = dict(zip(races["id"], races["competition.name"]))
    
    for race_index, (race_id, data) in enumerate(super_times.items(), start=1):
        df = data["data"]
        race_name = race_name_map.get(race_id, f"GP {race_index}")
        
        for _, row in df.iterrows():
            if by == "driver":
                name = row["driver.name"]
                image = row["driver.image"]
            else:
                name = row["team.name"]
                image = row["team.logo"]

            all_rows.append({
                "race_index": race_index,
                "race_name": race_name,
                "entity_name": name,
                "image_url": image,
                "time": row["time"],
                "time_ms": row["time_ms"],
                "superTimeRatio": row["superTimeRatio"],
                "superTimeDelta%": row["superTimeDelta%"],
            })

    return pd.DataFrame(all_rows)
