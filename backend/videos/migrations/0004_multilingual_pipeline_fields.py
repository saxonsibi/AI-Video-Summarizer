from django.db import migrations, models


def _backfill_multilingual_fields(apps, schema_editor):
    Transcript = apps.get_model('videos', 'Transcript')
    Summary = apps.get_model('videos', 'Summary')

    for tr in Transcript.objects.all().iterator():
        lang = (tr.language or 'en').strip() or 'en'
        updates = {
            'transcript_language': lang,
            'canonical_language': 'en' if lang != 'en' else lang,
            'transcript_original_text': tr.full_text or '',
            'transcript_canonical_text': tr.full_text or '',
        }
        Transcript.objects.filter(pk=tr.pk).update(**updates)

    for sm in Summary.objects.all().iterator():
        Summary.objects.filter(pk=sm.pk).update(
            summary_language='en',
            summary_source_language='en',
            translation_used=False,
        )


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0003_alter_video_youtube_url'),
    ]

    operations = [
        migrations.AddField(
            model_name='summary',
            name='summary_language',
            field=models.CharField(db_index=True, default='en', max_length=16),
        ),
        migrations.AddField(
            model_name='summary',
            name='summary_source_language',
            field=models.CharField(db_index=True, default='en', max_length=16),
        ),
        migrations.AddField(
            model_name='summary',
            name='translation_used',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='transcript',
            name='asr_engine',
            field=models.CharField(blank=True, default='faster_whisper', max_length=64),
        ),
        migrations.AddField(
            model_name='transcript',
            name='canonical_language',
            field=models.CharField(db_index=True, default='en', max_length=16),
        ),
        migrations.AddField(
            model_name='transcript',
            name='detection_confidence',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='transcript',
            name='script_type',
            field=models.CharField(blank=True, default='', max_length=32),
        ),
        migrations.AddField(
            model_name='transcript',
            name='transcript_canonical_text',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AddField(
            model_name='transcript',
            name='transcript_language',
            field=models.CharField(db_index=True, default='en', max_length=16),
        ),
        migrations.AddField(
            model_name='transcript',
            name='transcript_original_text',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.RunPython(_backfill_multilingual_fields, migrations.RunPython.noop),
    ]
