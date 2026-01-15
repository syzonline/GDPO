from enum import Enum
import airportsdata

iata_airports = airportsdata.load("IATA")  # key is the IATA location code :contentReference[oaicite:2]{index=2}

AIRPORT_IATA_LIST = [(code, code) for code in sorted(iata_airports.keys()) if code]
IATA = Enum("Airport", AIRPORT_IATA_LIST)
