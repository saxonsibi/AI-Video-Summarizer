from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0002_chatmessage_language_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='chatmessage',
            name='audio_url',
            field=models.CharField(blank=True, default='', max_length=500),
        ),
        migrations.AddField(
            model_name='chatmessage',
            name='voice_narration',
            field=models.TextField(blank=True, default=''),
        ),
    ]
