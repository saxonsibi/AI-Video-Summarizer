from django.db import migrations, models


def _backfill_hybrid_fields(apps, schema_editor):
    Transcript = apps.get_model('videos', 'Transcript')
    for tr in Transcript.objects.all().iterator():
        canon = (tr.transcript_canonical_text or tr.full_text or '').strip()
        eng = (tr.asr_engine or 'faster_whisper').strip() or 'faster_whisper'
        Transcript.objects.filter(pk=tr.pk).update(
            asr_engine_used=eng,
            transcript_quality_score=0.0,
            transcript_canonical_en_text=canon,
            canonical_language='en',
        )


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0004_multilingual_pipeline_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='transcript',
            name='asr_engine_used',
            field=models.CharField(blank=True, default='faster_whisper', max_length=64),
        ),
        migrations.AddField(
            model_name='transcript',
            name='transcript_canonical_en_text',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AddField(
            model_name='transcript',
            name='transcript_quality_score',
            field=models.FloatField(default=0.0),
        ),
        migrations.RunPython(_backfill_hybrid_fields, migrations.RunPython.noop),
    ]
