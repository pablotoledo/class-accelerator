import transformers
import torch

def summarize_llama3(text, prompt_value):
    model_id = "meta-llama/Meta-Llama-3.1-8B"
   
    try:
        # Configurar el tokenizador
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

        # Configurar el modelo con la configuración correcta de rope_scaling
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            rope_scaling={
                "type": "dynamic",
                "factor": 2.0
            }
        )

        # Crear el pipeline con el modelo y tokenizador configurados
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
       
        # Preparar el prompt
        full_prompt = f"Eres un asistente experto en resumir texto. Proporciona resúmenes concisos pero informativos.\n\n{prompt_value}\n\nTexto:\n{text}\n\nResumen:"

        # Generar el resumen
        output = pipeline(
            full_prompt,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=1
        )
       
        # Extraer el resumen generado
        if isinstance(output, list) and len(output) > 0:
            generated_text = output[0]['generated_text']
            summary = generated_text.split("Resumen:")[-1].strip()
        else:
            summary = "No se pudo generar un resumen coherente."
       
        return summary
    except Exception as e:
        print(f"Error al cargar el modelo o generar el resumen con Llama 3: {str(e)}")
        return None

# Texto de ejemplo para resumir
sample_text = """
El aprendizaje automático (machine learning) es una rama de la inteligencia artificial que se centra en el desarrollo de técnicas que permiten que las computadoras aprendan. En lugar de ser programadas explícitamente, estas máquinas son entrenadas usando grandes cantidades de datos y algoritmos que les dan la capacidad de aprender cómo realizar una tarea específica.
Hay varios tipos de aprendizaje automático, incluyendo el aprendizaje supervisado, no supervisado y por refuerzo. En el aprendizaje supervisado, el algoritmo se entrena en un conjunto de datos etiquetados, aprendiendo a mapear una entrada a una salida basándose en ejemplos de pares entrada-salida. El aprendizaje no supervisado, por otro lado, trata de encontrar patrones en datos no etiquetados.
Las aplicaciones del aprendizaje automático son vastas y variadas. Se utiliza en reconocimiento de voz, visión por computadora, filtros de spam, recomendaciones de productos, y muchas otras áreas. A medida que aumenta la cantidad de datos disponibles y mejora el poder de cómputo, el aprendizaje automático continúa avanzando y encontrando nuevas aplicaciones en diversos campos.
"""

# Prompt para el resumen
prompt = "Resuma el siguiente texto, identificando los puntos clave y ejemplos importantes. El resumen debe ser conciso pero informativo."

# Ejecutar la función de resumen
summary = summarize_llama3(sample_text, prompt)

# Imprimir el resumen
if summary:
    print("Resumen generado:")
    print(summary)
else:
    print("No se pudo generar el resumen.")