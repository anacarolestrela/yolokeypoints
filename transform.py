import os
from pathlib import Path

# ---------------- CONFIGURAÇÃO ----------------
# O CAMINHO RAIZ onde a busca deve começar.
# No seu caso, é a pasta 'frames'.
CAMINHO_RAIZ = r"C:\Users\anacarol\yolokeypoints\frames"

# Separador entre o nome da pasta e o número sequencial.
SEPARADOR = " - "

# ---------------- FUNÇÃO PRINCIPAL ----------------
def renomear_recursivamente(caminho_raiz):
    # 'os.walk' percorre a pasta raiz e todas as suas subpastas.
    for pasta_atual, _, arquivos in os.walk(caminho_raiz):
        # Ignora a pasta raiz, focando apenas nas subpastas que contêm arquivos.
        if pasta_atual == caminho_raiz:
            continue
        
        # 1. OBTER O NOME DA PASTA BASE:
        # Pega o último componente do caminho, que é o nome da subpasta.
        nome_pasta_base = Path(pasta_atual).name
        
        # 2. INICIAR O CONTADOR:
        contador = 1
        
        # 3. FILTRAR e ORDENAR os arquivos (boa prática para ordem sequencial)
        arquivos_ordenados = sorted(arquivos)

        for nome_arquivo_antigo in arquivos_ordenados:
            caminho_antigo_completo = os.path.join(pasta_atual, nome_arquivo_antigo)
            
            # Garante que estamos lidando apenas com arquivos (e não pastas ocultas ou links simbólicos)
            if not os.path.isfile(caminho_antigo_completo):
                continue
                
            # 4. SEPARAR NOME e EXTENSÃO:
            # os.path.splitext separa "nome.ext" em ("nome", ".ext")
            _, extensao = os.path.splitext(nome_arquivo_antigo)
            
            # 5. CRIAR O NOVO NOME:
            # Formata o contador com zero à esquerda (D2)
            novo_nome_base = f"{nome_pasta_base}{SEPARADOR}{contador:02d}{extensao}"
            caminho_novo_completo = os.path.join(pasta_atual, novo_nome_base)

            # 6. RENOMEAR:
            try:
                os.rename(caminho_antigo_completo, caminho_novo_completo)
                print(f"Renomeado em '{nome_pasta_base}': '{nome_arquivo_antigo}' -> '{novo_nome_base}'")
                contador += 1
            except Exception as e:
                print(f"ERRO ao renomear {nome_arquivo_antigo}: {e}")

# ---------------- EXECUÇÃO ----------------
if __name__ == "__main__":
    print(f"Iniciando renomeação recursiva a partir de: {CAMINHO_RAIZ}\n")
    renomear_recursivamente(CAMINHO_RAIZ)
    print("\nProcesso concluído.")