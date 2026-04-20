from backend.model import load_model
from frontend.ui import run_app

model, acc = load_model()
run_app(model, acc)