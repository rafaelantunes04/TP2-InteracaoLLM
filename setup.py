import os
import shutil
import requests
import tarfile
from tqdm import tqdm

if __name__ == "__main__":
    url = "https://drive.usercontent.google.com/download?id=1iq93lCdhaPUN0fWbLieMtzfB1850pKwd&confirm=t&uuid=cfad9241-60af-4aca-b116-6d619b4e2e34&at=AAINaIJGx5MdjCUmNC6b8EaIb87U%3A1780770040092"
    diretorio_destino = "./data/"
    caminho_ficheiro = os.path.join(diretorio_destino, "dataset.tar.gz")

    # Perguntar ao utilizador antes de fazer download
    resposta_user = input("O ficheiro tem cerca de 11.4 GB. Queres iniciar o download? (s/n): ")

    if resposta_user.lower().strip() in ['s', 'sim', 'y']:
        os.makedirs(diretorio_destino, exist_ok=True)

        print("A iniciar a transferência...")
        resposta = requests.get(url, stream=True)

        tamanho_total = int(resposta.headers.get('content-length', 0))

        # Download com barra de progresso e percentagem
        with open(caminho_ficheiro, 'wb') as ficheiro, tqdm(
            desc="A transferir",
            total=tamanho_total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as barra_progresso:
            for pedaco in resposta.iter_content(chunk_size=8192):
                if pedaco:
                    tamanho = ficheiro.write(pedaco)
                    barra_progresso.update(len(pedaco))

        print("\nDownload concluído! A extrair os ficheiros...")

        # Extrair o ficheiro comprimido
        with tarfile.open(caminho_ficheiro, "r:gz") as tar:
            tar.extractall(path=diretorio_destino)

        os.remove(caminho_ficheiro)

        # Reorganizar a estrutura de pastas
        print("A reorganizar os ficheiros...")

        pasta_extraida = "./data/SKU110K_fixed"

        # Mover ./data/SKU110K_fixed/images -> ./data/images/
        shutil.move(os.path.join(pasta_extraida, "images"), "./data/images")

        # Mover ./data/SKU110K_fixed/LICENSE.txt -> ./data/LICENSE.txt
        shutil.move(os.path.join(pasta_extraida, "LICENSE.txt"), "./data/LICENSE.txt")

        # Apagar a pasta SKU110K_fixed e tudo o que restar dentro dela
        shutil.rmtree(pasta_extraida)

        print("Processo finalizado com sucesso!")

    else:
        print("Download cancelado.")