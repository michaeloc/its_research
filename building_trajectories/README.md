# Classe Preprocess

- Organiza os dados brutos em trajetórias
- Elimina dados/trajetórias incompletas
- Gera novas informações
- Armazena trajetórias

## FONTE do dataset: Dublin Bus GPS sample data from Dublin City Council (Insight Project)

Bus GPS Data Dublin Bus GPS data across Dublin City, from Dublin **City Council'traffic control**, in csv format. 

Each datapoint (row in the CSV file) has the following entries:

- Timestamp micro since 1970 01 01 00:00:00 GMT
- Line ID
- Direction
- Journey Pattern ID
- Time Frame (The start date of the production time table - in Dublin the production time table starts at 6am and ends at 3am)
- Vehicle Journey ID (A given run on the journey pattern)
- Operator (Bus operator, not the driver)
- Congestion [0=no,1=yes]
- Lon WGS84
- Lat WGS
- Delay (seconds, negative if bus is ahead of schedule)
- Block ID (a section ID of the journey pattern)
- Vehicle ID
- Stop ID
- At Stop [0=no,1=yes]

## Hipótose sobre unicidade das trajetórias

Destes os seguintes atributos nos indicam uma trajetória única
(segrega também trechos diferentes de uma mesma linha, ex: ida e
volta)

``` python
trajetoria = ['line_id','journey_id','time_frame','vehicle_journey_id','operator','vehicle_id']
```

## Limpeza de outliers

### Técnicas utilizadas / # pontos/trajetórias filtradas

## Schema DB
