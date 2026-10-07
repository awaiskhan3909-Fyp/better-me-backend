import gradio as gr
from app.main import app as fastapi_app

# Mount FastAPI app onto Gradio
demo = gr.mount_gradio_app(fastapi_app, gr.Blocks(title="Better Me API Backend"), path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)
