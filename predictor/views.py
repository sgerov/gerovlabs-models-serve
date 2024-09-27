from django.shortcuts import render

import base64
import json
import os
import subprocess
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from fastai.vision.all import *

@csrf_exempt  # Disable CSRF for simplicity (not recommended for production)
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_base64 = data.get('image')

            if not image_base64:
                return JsonResponse({'error': 'No image provided'}, status=400)

            # Decode base64 image
            image_data = image_base64.replace('data:image/png;base64,', '')
            image_path = 'input_image.png'
            with open(image_path, 'wb') as img_file:
                img_file.write(base64.b64decode(image_data))

            # Run the FastAI inference script
            result = subprocess.run(['python', 'infer.py', image_path], capture_output=True, text=True)

            model = load_learner('model.pkl')
            img = PILImage.create(image_path)
            _, _, outputs = model.predict(img)

            result = {
                'predictions': dict(zip(model.dls.vocab, map(float, outputs))),
            }

            return JsonResponse(result)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid HTTP method'}, status=405)

