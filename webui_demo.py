import gradio as gr

def greet(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "checkbox", gr.Slider(0, 100)],
    outputs=["text", "number"],
)


from transformers import pipeline

pipe = pipeline("image-classification")

demo2 = gr.Interface.from_pipeline(pipe).launch()


if __name__ == "__main__":
    # demo.launch(share=True)

    demo2.launch()