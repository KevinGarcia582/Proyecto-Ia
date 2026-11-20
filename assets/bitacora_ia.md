# Bitácora de Vibe Coding

Este archivo resume el uso de asistentes de IA durante la implementación y revisión del prototipo. Las salidas deben contrastarse con el código, los notebooks y la ejecución local. No se archivaron capturas de las conversaciones de apoyo; por eso no se presentan capturas inexistentes.

| Fecha | Herramienta | Apoyo recibido | Verificación y resultado | Evidencia |
|---|---|---|---|---|
| 2026-11-18 | ChatGPT/Codex | Apoyo para estructurar el prototipo de análisis de reseñas, la red neuronal y el agente con herramientas. | Se revisaron los archivos de la aplicación y los resultados guardados en los notebooks. | Captura de conversación no archivada. |
| 2026-11-20 | ChatGPT/Codex | Revisión de rutas, documentación y manejo de la entrada vacía en la predicción directa. | Se comprobó la compilación de los módulos Python y se ejecutó la inferencia local con ejemplos positivos y negativos. La aplicación ahora muestra una advertencia si se intenta clasificar un campo vacío. | Captura de conversación no archivada. |

## Hallazgo técnico

Durante la evaluación del agente apareció un aviso del SDK relacionado con el parámetro temperature: el modelo podía usar valores predeterminados e ignorar ese parámetro. Esto afecta la posibilidad de reproducir exactamente las respuestas generativas. La clasificación del sentimiento se ejecuta localmente con los pesos entrenados y no depende de esa configuración.
