
import pandas as pd
import os

if __name__ == "__main__":
    # 메타데이터 전처리한 csv 불러오기
    foreign_csv = pd.read_csv('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/data/foregin_meta_csv.csv', encoding='utf-8')
    kor_csv = pd.read_csv('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/data/kor_meta_csv.csv', encoding='utf-8')

    # 국가 잘못 처리 한 것 수정 및 학습기간이 24개월 이상만 사용
    foreign_csv['county'] = foreign_csv['audio_path'].map(lambda x : x.split('\\')[-1][:2])
    foreign_csv = foreign_csv[foreign_csv['learning_period'] >= 24]

    # 학습 데이터 셋 비율을 3:1로 맞추기 위해 한국인 데이터셋에서 5만개 랜덤 추출
    random_kor = kor_csv.sample(n=50000, random_state=23)
    random_kor['county'] = 'KR'

    # 사용할 데이터 셋에 대한 최종 메타데이터 정리 csv저장
    final_dataset_df = pd.concat([foreign_csv, random_kor], ignore_index=True)
    final_dataset_df = final_dataset_df.rename(columns={'county':'country'})

    os.chdir('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/data')
    final_dataset_df.to_csv('./final_dataset_df.csv', index=False, encoding='utf-8')

    print('-------------finish-------------')