import boto3

with open('face_image.jpg', 'rb') as image_file:
    image = image_file.read()

client = boto3.client('rekognition')

response = client.detect_faces(
    Image={
        'Bytes': image
    },
    Attributes=['ALL']
)

for face in response['FaceDetails']:
    # 表情の分類結果を取得
    emotions = face['Emotions']
    
    # 最も確率の高い表情を取得
    max_emotion = max(emotions, key=lambda x: x['Confidence'])
    emotion_name = max_emotion['Type']
    emotion_confidence = max_emotion['Confidence']
    
    # 分類結果の表示
    print('Emotion: {}, Confidence: {}'.format(emotion_name, emotion_confidence))