from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='chatmessage',
            name='output_language',
            field=models.CharField(default='en', max_length=16),
        ),
        migrations.AddField(
            model_name='chatmessage',
            name='retrieval_language',
            field=models.CharField(default='en', max_length=16),
        ),
        migrations.AddField(
            model_name='chatmessage',
            name='user_language',
            field=models.CharField(default='en', max_length=16),
        ),
    ]
