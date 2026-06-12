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
        pasta_imagens = "./data/images"

        # Mover ./data/SKU110K_fixed/images -> ./data/images/
        shutil.move(os.path.join(pasta_extraida, "images"), pasta_imagens)

        # Mover ./data/SKU110K_fixed/LICENSE.txt -> ./data/LICENSE.txt
        shutil.move(os.path.join(pasta_extraida, "LICENSE.txt"), "./data/LICENSE.txt")

        # Apagar a pasta SKU110K_fixed e tudo o que restar dentro dela
        shutil.rmtree(pasta_extraida)

        # Manter apenas as primeiras 500 fotos de teste e apagar TODAS as outras
        print("A limpar as fotos excedentes...")
        ficheiros_imagens = sorted(os.listdir(pasta_imagens))
        
        # Filtra apenas os ficheiros que contêm "test" no nome
        fotos_teste = [f for f in ficheiros_imagens if "test" in f.lower()]
        
        # Guardar a lista das 500 em ordem para o processo de renomear
        lista_500_fotos = fotos_teste[:500]
        
        # Define as 500 que queremos guardar num conjunto (set) para verificação mais rápida ao apagar
        fotos_para_guardar = set(lista_500_fotos)
        
        # Apaga tudo o que não estiver na lista das 500 escolhidas (incluindo train_xx.jpg e val_xx.jpg)
        for foto in ficheiros_imagens:
            if foto not in fotos_para_guardar:
                os.remove(os.path.join(pasta_imagens, foto))

        # Renomear as 500 imagens mantidas
        print("A renomear as imagens guardadas...")
        for indice, nome_antigo in enumerate(lista_500_fotos, start=1):
            caminho_antigo = os.path.join(pasta_imagens, nome_antigo)
            
            # Extrai a extensão original (ex: '.jpg') para manter a integridade do ficheiro
            extensao = os.path.splitext(nome_antigo)[1] 
            novo_nome = f"img_{indice}{extensao}"
            caminho_novo = os.path.join(pasta_imagens, novo_nome)
            
            os.rename(caminho_antigo, caminho_novo)

        print("Processo finalizado com sucesso!")

    else:
        print("Download cancelado.")