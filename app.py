from app.gradio_app import demo  # import your Gradio interface

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True   # 🔥 this creates a public Gradio link
    )
