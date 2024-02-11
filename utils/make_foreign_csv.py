import json
from G2P.KoG2Padvanced import KoG2Padvanced
import re
from tqdm import tqdm
from glob import glob
import pandas as pd
import os


# 두번째 'ㅇ'이 들어가는 글자에 경우 모음만 나와서 이를 수정하는 함수
def vowel_change(hangul_string):
    vowels = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
                       'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    vowels_change = ['아', '애', '야', '얘', '어', '에', '여', '예', '오', '와',
                     '왜', '외', '요', '우', '워', '웨', '위', '유', '으', '의', '이']
    dic = {i:k for i, k in zip(vowels,vowels_change)}
    
    pattern = "[" + "".join(vowels) + "]"
    result = re.sub(pattern, lambda x: dic[x.group(0)], hangul_string)
    return result

# 텍스트 g2p 변환을 위한 함수
def foreign_g2p(text):
    text = re.sub('[^가-힣\s]', '', text)
    text = re.sub('\s+', ' ', text)
    text = KoG2Padvanced(text)
    text = vowel_change(text)
    return text



if __name__ == "__main__":

    # 외국인 및 말하기 평가 셋이 있는 경로 지정
    path_foreign =  "C:\\Users\\jjw28\\Downloads\\131.인공지능 학습을 위한 외국인 한국어 발화 음성 데이터\\01.데이터_new_20220719\\2.Validation\\라벨링데이터"

    # 각 경로에 있는 라벨 데이터 확인
    foreign_json_list = glob(path_foreign + '/**/**/*.json')

    # 영어가 있는 텍스트를 제외하기 위한 패턴 지정
    pattern = re.compile(r'[A-Za-z]')

    # 한 번에 데이터프레임에 넣기 위해 임시 리스트 생성
    tmp_list = []
    
    # 외국인 json파일 경로가 있는 리스트 순회
    for foreign_json in tqdm(foreign_json_list, desc='to_csv..'):

        # 각 음성에 대한 JSON 파일 불러오기
        with open(foreign_json, 'r', encoding='utf-8') as jsf:
            data = json.load(jsf)

        # 필요한 값들 JSON에서 추출
        audio_path = foreign_json.replace('라벨링데이터', '원천데이터').replace('json', 'wav')
        if data['transcription'].get('AnswerLabelText', False):
            text = data['transcription'].get('AnswerLabelText', False)
        else:
            text = data['transcription']['ReadingLabelText']

        # 영어가 포함되면 데이터 셋에 포함되지 않게 다음으로 넘어감
        if bool(pattern.search(text)):
            continue

        g2p_text = foreign_g2p(text)
        country = data['residence_info']['country']
        learning_period = data['skill_info']['LearningPeriod']
        record_time = data['file_info']['recordTime']

        tmp_list.append([audio_path, text, g2p_text, country, learning_period, record_time])
    
    # 기본 경로 변경
    os.chdir('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/data')
             
    # 외국인 음성 데이터에 대한 전체 경로 및 메타데이터 저장을 위한 데이터프레임
    foreign_meta_csv = pd.DataFrame(tmp_list, columns = ['audio_path', 'text', 'g2p_text','county', 'learning_period', 'record_time'])
    foreign_meta_csv.to_csv('./foregin_meta_csv.csv', index = False, encoding = 'utf-8')

    print('-------------finish-------------')
