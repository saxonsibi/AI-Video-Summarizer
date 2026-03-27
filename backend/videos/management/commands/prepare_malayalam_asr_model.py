from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from videos.utils import _ensure_malayalam_ctranslate2_model


class Command(BaseCommand):
    help = "Pre-convert the configured Malayalam Transformers Whisper model into a cached CTranslate2 model directory."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model",
            dest="model",
            default="",
            help="Optional Malayalam model override. Defaults to ASR_MALAYALAM_PRIMARY_MODEL.",
        )
        parser.add_argument(
            "--compute-type",
            dest="compute_type",
            default="",
            help="Optional quantization override. Defaults to ASR_MALAYALAM_COMPUTE_TYPE.",
        )

    def handle(self, *args, **options):
        model_name = str(options.get("model") or getattr(settings, "ASR_MALAYALAM_PRIMARY_MODEL", "") or "").strip()
        compute_type = str(options.get("compute_type") or getattr(settings, "ASR_MALAYALAM_COMPUTE_TYPE", "int8") or "int8").strip().lower()
        if not model_name:
            raise CommandError("No Malayalam model configured. Set ASR_MALAYALAM_PRIMARY_MODEL or pass --model.")

        self.stdout.write(f"Preparing Malayalam model: {model_name}")
        self.stdout.write(f"Compute type: {compute_type}")
        try:
            resolved_model, meta = _ensure_malayalam_ctranslate2_model(model_name, compute_type)
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS("Malayalam model preparation completed."))
        self.stdout.write(f"Configured model: {meta.get('configured_model_name', model_name)}")
        self.stdout.write(f"Configured family: {meta.get('model_family', 'auto')}")
        self.stdout.write(f"Converted model path: {resolved_model}")
        self.stdout.write(f"Converted model valid: {meta.get('converted_model_valid', False)}")
        self.stdout.write("")
        self.stdout.write("Recommended production settings:")
        self.stdout.write(f"ASR_MALAYALAM_PRIMARY_MODEL={resolved_model}")
        self.stdout.write("ASR_MALAYALAM_MODEL_FAMILY=ctranslate2")
