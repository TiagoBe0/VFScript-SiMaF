# VFScript-SiMaF

## 📄 Ejemplo completo de `input_params.json`

```json
{
    "CONFIG": [
        {   "training":true,
            "geometric_method":false,
            "activate_generate_relax": true,
            "generate_relax": [
                "bcc",
                "3.4359",
                10,10,10,
                "Fe"
            ],
            "relax": "inputs/relax_structure.dump",
            "defect": [

                "inputs/fe_2v"
            ],
            "radius": 2,
            "smoothing_level": 0,
            "smoothing_level_training": 0,
            "max_graph_size": 15,
            "max_graph_variations": 10,
            
            "cutoff": 3,
            "radius_training": 5,
            "training_file_index":15,
            "cluster tolerance": 2,
            "divisions_of_cluster": 6,
            "iteraciones_clusterig": 4
        }
    ],
    "PREDICTOR_COLUMNS": [
        "surface_area",
        "filled_volume",
        "cluster_size"
    ]
}

📌 Instrucciones clave

    El archivo debe llamarse exactamente: input_params.json

    Debe estar ubicado en el mismo directorio desde donde ejecutás el programa.

    Si el archivo no se encuentra, el programa mostrará un error como:

FileNotFoundError: No se encontró 'input_params.json' en: ...

🧪 Ejemplo de ejecución

Si tu archivo Python se llama myfile.py, asegurate de que la estructura sea:

tu_proyecto/
├── input_params.json
└── myfile.py

Y ejecutás con:

python myfile.py




