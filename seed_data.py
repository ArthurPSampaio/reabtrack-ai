import requests
import random
from datetime import datetime, timedelta
import time

# Configuração
API_URL = "http://localhost:3000"  # URL do NestJS

# Dados Auxiliares
GENEROS = ["Masculino", "Feminino"]
TIPOS_ATIVIDADE = ["Fortalecimento", "Alongamento", "Aeróbico", "Equilíbrio", "Outro"]

def log(msg):
    print(f"[SEED] {msg}")

def create_paciente(nome, nascimento, genero, diag, sintomas):
    payload = {
        "nome": nome,
        "dataNascimento": nascimento,
        "genero": genero,
        "diagnostico": diag,
        "sintomas": sintomas
    }
    r = requests.post(f"{API_URL}/pacientes", json=payload)
    if r.status_code == 201:
        return r.json()
    else:
        print(f"Erro ao criar paciente: {r.text}")
        return None

def create_plano(paciente_id, objetivo, diag_rel):
    payload = {
        "pacienteId": paciente_id,
        "objetivoGeral": objetivo,
        "diagnosticoRelacionado": diag_rel,
        "status": "Ativo",
        "dataFimPrevista": (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d"),
        "atividades": [
            {
                "nome": "Exercício Base A",
                "descricao": "Executar com calma",
                "tipo": "Fortalecimento",
                "series": 3, "repeticoes": 10, "frequencia": "diario"
            },
            {
                "nome": "Exercício Base B", 
                "tipo": "Alongamento",
                "series": 2, "repeticoes": 30, "frequencia": "diario"
            }
        ]
    }
    r = requests.post(f"{API_URL}/planos", json=payload)
    if r.status_code == 201:
        return r.json()
    return None

def create_registro(paciente_id, plano_id, data_sessao, dor, esforco, subjetiva, objetiva, avaliacao):
    payload = {
        "pacienteId": paciente_id,
        "planoId": plano_id,
        "dataSessao": data_sessao,
        "escalaDor": dor,
        "percepcaoEsforco": esforco,
        "conseguiuRealizarTudo": True,
        "notasSubjetivas": subjetiva,
        "notasObjetivas": objetiva,
        "avaliacao": avaliacao,
        "planoProximaSessao": "Manter e progredir carga."
    }
    r = requests.post(f"{API_URL}/registros", json=payload)
    if r.status_code == 201:
        print(f"   -> Registro criado: Dor {dor}/10 em {data_sessao}")
        # Pequena pausa para garantir que o NestJS emita o evento e o Python processe
        # Isso evita condições de corrida no FAISS se o computador for lento
        time.sleep(0.2) 
    else:
        print(f"Erro registro: {r.text}")

# =================================================================================
# CENÁRIOS CLÍNICOS
# =================================================================================

def gerar_cenario_joao_ombro():
    log("Gerando João (Ombro - Recuperação Lenta)...")
    p = create_paciente("João Silva", "1985-05-20", "Masculino", "Tendinite Manguito Rotador", "Dor aguda ao elevar braço")
    plano = create_plano(p["id"], "Restaurar ADM completa sem dor", "Tendinite Supraespinhal")
    
    # Sessão 1 a 10 (Evolução)
    base_date = datetime.now() - timedelta(days=60)
    for i in range(10):
        data = (base_date + timedelta(days=i*5)).strftime("%Y-%m-%dT10:00:00.000Z")
        
        # Lógica de evolução: Dor cai, Esforço sobe
        dor = max(1, 8 - int(i * 0.7)) # Começa 8, termina ~1
        esforco = min(10, 3 + int(i * 0.6)) # Começa 3, termina ~9
        
        subjetiva = f"Paciente relata {'muita' if dor > 5 else 'pouca'} dor hoje."
        objetiva = f"ADM de flexão {'limitada a 90°' if i < 4 else 'completa 180°'}. Força grau {3 if i < 5 else 5}."
        avaliacao = "Evolução positiva." if i > 0 else "Avaliação inicial."

        create_registro(p["id"], plano["id"], data, dor, esforco, subjetiva, objetiva, avaliacao)

def gerar_cenario_maria_lca():
    log("Gerando Maria (LCA - Recuperação Rápida)...")
    p = create_paciente("Maria Oliveira", "1995-03-12", "Feminino", "Pós-op LCA Joelho", "Instabilidade e edema")
    plano = create_plano(p["id"], "Ganho de força quadríceps e propriocepção", "LCA")
    
    base_date = datetime.now() - timedelta(days=45)
    for i in range(8):
        data = (base_date + timedelta(days=i*5)).strftime("%Y-%m-%dT14:00:00.000Z")
        
        dor = max(0, 6 - i) # Recuperação rápida
        esforco = 5 + (i % 3) # Esforço constante
        
        subjetiva = "Sente o joelho mais firme."
        objetiva = "Sem edema visível. Cicatriz ok."
        avaliacao = "Ótima resposta ao treino de força."

        create_registro(p["id"], plano["id"], data, dor, esforco, subjetiva, objetiva, avaliacao)

def gerar_cenario_roberto_lombar():
    log("Gerando Roberto (Lombalgia - Oscilante)...")
    p = create_paciente("Roberto Santos", "1970-11-05", "Masculino", "Lombalgia Crônica", "Dor ao ficar sentado")
    plano = create_plano(p["id"], "Core strengthening e mobilidade", "Hérnia L4-L5")
    
    base_date = datetime.now() - timedelta(days=30)
    dores = [7, 6, 8, 5, 4, 7, 3] # Oscilação (teve recaídas)
    
    for i, dor in enumerate(dores):
        data = (base_date + timedelta(days=i*4)).strftime("%Y-%m-%dT09:00:00.000Z")
        esforco = 4
        
        if dor > 6:
            sub = "Relatou crise de dor após trabalho."
            av = "Realizada analgesia e mobilização leve."
        else:
            sub = "Chegou bem melhor hoje."
            av = "Foco em fortalecimento de core."

        create_registro(p["id"], plano["id"], data, dor, esforco, sub, "Teste de Lasègue negativo.", av)

def gerar_cenario_julia_tornozelo():
    log("Gerando Julia (Entorse - Recente)...")
    p = create_paciente("Julia Costa", "2000-01-15", "Feminino", "Entorse Tornozelo Grau 2", "Inchaço e dor ao pisar")
    plano = create_plano(p["id"], "Reduzir edema e treino de marcha", "Entorse Inversão")
    
    # Apenas 2 sessões
    base_date = datetime.now() - timedelta(days=3)
    
    # Sessão 1
    create_registro(p["id"], plano["id"], base_date.strftime("%Y-%m-%dT16:00:00.000Z"), 
                    9, 2, "Não consegue pisar.", "Edema ++.", "Protocolo PRICE.")
    
    # Sessão 2
    create_registro(p["id"], plano["id"], datetime.now().strftime("%Y-%m-%dT16:00:00.000Z"), 
                    7, 3, "Um pouco melhor, pisa com muleta.", "Edema +.", "Iniciado mobilização passiva.")

if __name__ == "__main__":
    print("--- INICIANDO POPULAÇÃO DO BANCO & INGESTÃO DE IA ---")
    try:
        gerar_cenario_joao_ombro()
        gerar_cenario_maria_lca()
        gerar_cenario_roberto_lombar()
        gerar_cenario_julia_tornozelo()
        print("\n--- SUCESSO! DADOS CRIADOS E ENVIADOS PARA A IA ---")
        print("Agora abra o app e gere os relatórios.")
    except Exception as e:
        print(f"ERRO CRÍTICO: {e}")
        print("Verifique se o backend NestJS está rodando na porta 3000.")