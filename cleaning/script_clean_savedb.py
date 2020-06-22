#!/usr/bin/python
import psycopg2 as ps
import datetime as dt
from tqdm import tqdm
import preprocess as prep
import pandas as pd

def save_in_db(data):
    conn = ps.connect("dbname=urbanmobility user=postgres password=242124")
    cur = conn.cursor()
    grouped = data.groupby('matricula')
    for name, item in tqdm(grouped):
        all_data = data[data['matricula']==name]
        cur.execute("INSERT INTO linha2 (matricula, unidade, nome, estado, linha) VALUES (%s, %s, %s, %s, %s)",
            (all_data.matricula.values[0], all_data.Unidad.values[0],
            all_data.nombre.values[0], all_data.Estado.values[0],
            all_data.Linea.values[0]))
        conn.commit()
        for i in range(len(all_data)):
            time = dt.datetime.strptime(all_data.Instante.values[i],'%Y-%m-%d %H:%M:%S.%f')
            cur.execute("INSERT INTO pontos2 (lat, lng, instante, rota, posicao, velocidade,viaje, matricula_id) VALUES (%s, %s, %s, %s, %s, %s,%s, %s)",
            (all_data.lat.values[i],all_data.lng.values[i],time,all_data.Ruta.values[i],all_data.Posicion.values[i],all_data.Velocidad.values[i],all_data.Viaje.values[i],all_data.matricula.values[i]))
            conn.commit()


if __name__ == "__main__":

    conn = ps.connect("dbname=urbanmobility user=postgres password=242124")
    cur = conn.cursor()
    print('Criando banco de dados...')
    cur.execute("CREATE TABLE linha2 (matricula varchar PRIMARY KEY, unidade numeric, nome varchar, estado numeric,linha numeric );")
    cur.execute("CREATE TABLE pontos2 (id serial PRIMARY KEY, lat numeric, lng numeric, instante timestamp, rota numeric, posicao numeric, velocidade numeric,viaje numeric, matricula_id varchar );")
    conn.commit()


    data_frame = pd.read_csv('011017_011117.csv',sep=';')
    print('Fazendo o cleaning dos dados...')
    new_data = prep.PreProcess().clean_data(data_frame)
    print('Convertendo coordenadas para latitude e longitude')
    lat_lng = prep.PreProcess().coordinates_to_latlng(new_data)
    print('Salvando os dados no banco de dados....')
    save_in_db(lat_lng)
