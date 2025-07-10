import logging

def configurar_logger(nombre_archivo="vectorstore_log.txt"):
    logging.basicConfig(
        filename=nombre_archivo,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger()