from rest_framework import serializers
from .models import UploadedImage

class TextSerializer(serializers.Serializer):
    text = serializers.CharField()

class UploadedImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedImage
        fields = ['image']
        # fields = ('image', )