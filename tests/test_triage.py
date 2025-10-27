import io
import sys
import os
from PIL import Image

# Ensure the project root is on sys.path so `from app import app` works when
# running this script directly from tests/ directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app


def make_test_image():
    img = Image.new('RGB', (224, 224), color='white')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf


if __name__ == '__main__':
    buf = make_test_image()
    data = {'image': (buf, 'test.jpg')}
    with app.test_client() as c:
        resp = c.post('/triage', data=data, content_type='multipart/form-data')
        print('STATUS', resp.status_code)
        try:
            print(resp.get_json())
        except Exception:
            print(resp.data)
