from django.shortcuts import render

import base64
import json
import os
import subprocess
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tempfile

from fastai.vision.all import *

# Cache the model to avoid reloading it in every request
model = load_learner('model.pkl')

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

            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_img_file:
                temp_img_file.write(base64.b64decode(image_data))
                temp_img_file.flush()  # Ensure the file is written

                # Load the model and make a prediction
                img = PILImage.create(temp_img_file.name)
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