import requests
import pandas as pd
from datetime import datetime
import time


def fetch_weather_data(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # 检查请求是否成功
    data = response.json()
    return data


def parse_weather_data(data):
    # 提取 metadata 中的 count 字段
    metadata_count = data['metadata']['collection']['count']

    # 提取 collection 中的天气数据
    records = data['collection']
    parsed_records = []

    for record in records:
        #         print(record)
        parsed_record = {
            'dateTime': datetime.strptime(record['dateTime'], "%Y-%m-%dT%H:%M:%SZ"),
            'airTemperature': record['airTemperature'],
            'relativeHumidity': record['relativeHumidity'],
            'soilTemperature': record['soilTemperature'],
            'solarIrradiance': record['solarIrradiance'],
            'rainfall': record['rainfall'],
            'dewPoint': record['dewPoint'],
            'deltaT': record['deltaT'],
            'wetBulb': record['wetBulb'],
            'batteryVoltage': record['batteryVoltage'],
            'wind_speed': record['wind'][0]['avg']['speed'],
            'wind_direction': record['wind'][0]['avg']['direction']['compassPoint'],
            'wind_degrees': record['wind'][0]['avg']['direction']['degrees'],
        }
        parsed_records.append(parsed_record)

    df = pd.DataFrame(parsed_records)
    return df, metadata_count


def get_all_weather_data(station_id, start_date_time, end_date_time, headers, limit=200):
    base_url = 'https://api.agric.wa.gov.au/v2/weather/stations/{}/data?'.format(station_id)
    all_data = pd.DataFrame()
    offset = 0

    while True:
        url = f"{base_url}startDateTime={start_date_time}&endDateTime={end_date_time}&offset={offset}&limit={limit}&sort=-dateTime"
        data = fetch_weather_data(url, headers)
        df, count = parse_weather_data(data)

        if df.empty:
            break

        all_data = pd.concat([all_data, df], ignore_index=True)
        offset += limit

        if offset >= count:
            break

        # 睡眠 0~1 秒
        time.sleep(max(0.8, min(1, 0.8 + (0.5 - time.time() % 1))))

    return all_data


# 示例使用
station_id = 'WN'
start_date_time = '2024-03-19T00:00:00'
end_date_time = '2025-08-01T00:00:00'

headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'en-GB,en;q=0.9,en-US;q=0.8,en-AU;q=0.7',
    'Access-Control-Max-Age': '600',
    'Connection': 'keep-alive',
    'Host': 'api.agric.wa.gov.au',
    'Origin': 'https://weather.agric.wa.gov.au',
    'Referer': 'https://weather.agric.wa.gov.au/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0',
    'api_key': '28FCCD2FAAFEFA6DDC8EAD73.apikey',
    'sec-ch-ua': '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"'
}

all_weather_data = get_all_weather_data(station_id, start_date_time, end_date_time, headers)

if not all_weather_data.empty:
    print(all_weather_data.head(10))
    all_weather_data.to_csv(
        '{0}_all_weather_data_from_{1}_to_{2}.csv'.format(station_id, start_date_time, end_date_time), index=False)
else:
    print("没有数据返回")