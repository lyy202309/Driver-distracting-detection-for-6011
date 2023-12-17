import base64
import matplotlib.image as mpimg
import requests
import matplotlib.pyplot as plt


def get_public_ip():
   return requests.get('http://ip.42.pl/raw').text

def get_location(ip, key):
    url = f'https://restapi.amap.com/v3/ip?ip={ip}&output=json&key={key}'
    data = requests.get(url).json()
    return data




def detect():

    url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/" \
          "driver_behavior?access_token=24.12c17ce7b3942d419767b200" \
          "39d61751.2592000.1704123922.282335-44189270"
    image_file = 'smoke.jpg'
    img_show = mpimg.imread('smoke.jpg')
    with open(image_file, 'rb') as f:
        image_data = f.read()
    img=base64.b64encode(image_data)
    params={"image":img}
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    response = requests.post(url, headers=headers, data=params)
    ip = get_public_ip()
    location = get_location(ip, '7cf01841d11f9e10ccc014b9d4ac0ee2')


    if response:
        data = response.json()
        attributes = data['person_info'][0]['attributes']
        if 'no_face_mask' in attributes:
            del attributes['no_face_mask']
        labels = list(attributes.keys())
        scores = [attribute['score'] for attribute in attributes.values()]



        fig=plt.figure(figsize=(10, 5))
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(img_show)
        ax1.axis('off')

        ax2 = plt.subplot(1, 2, 2)
        y_pos = range(len(labels))

        rects = ax2.barh(y_pos, scores, align='center')
        for rect, score in zip(rects, scores):
            if score > 0.5:
                rect.set_facecolor('red')
            else:
                rect.set_facecolor('blue')

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.invert_yaxis()
        ax2.set_xlabel('Score')
        ax2.set_title('Driver Behavior Attributes')
        location['location'] = location['rectangle']
        location_text=location['location']


        ax2.text(0.5, -0.2, f"Location: {location_text}", transform=ax2.transAxes, ha='center', va='center')
        plt.tight_layout()
        plt.show()





if __name__ == '__main__':

 detect()

