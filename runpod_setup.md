# Guide de déploiement RunPod pour Qwen Coder v2.5 7B LoRA

## 1. Préparer votre projet localement

```bash
# Commit vos changements
git add .
git commit -m "Update for Qwen Coder v2.5 7B training"
git push
```

## 2. Créer un pod RunPod

1. Allez sur [RunPod.io](https://runpod.io)
2. Créez un nouveau pod avec :
   - **GPU**: RTX 4090 ou A100 (minimum 24GB VRAM pour 7B model)
   - **Template**: PyTorch 2.1 + CUDA 12.1
   - **Disk**: 50GB minimum
   - **Options**: SSH activé

## 3. Se connecter au pod

```bash
ssh root@[YOUR_POD_IP] -p [YOUR_SSH_PORT]
```

## 4. Cloner votre projet

```bash
cd /workspace
git clone [YOUR_GITHUB_REPO_URL]
cd EndToEndLoRA
```

## 5. Configurer HuggingFace

```bash
# Obtenir un token depuis https://huggingface.co/settings/tokens
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
```

## 6. Lancer le script de déploiement

```bash
./runpod_deploy.sh
```

## 7. Lancer l'entraînement

```bash
uv run train.py
```

## 8. Surveiller l'entraînement

Dans un autre terminal SSH :
```bash
watch -n 1 nvidia-smi
```

## Paramètres d'entraînement optimisés

Le script est configuré pour :
- **Batch size**: 1 (pour économiser la VRAM)
- **Gradient accumulation**: 4 steps
- **Max steps**: 100 (ajustable)
- **Learning rate**: 2e-4
- **Mixed precision**: FP16
- **8-bit optimizer**: Pour économiser la mémoire

## Après l'entraînement

Les modèles seront sauvegardés dans :
- `Qwen2.5-Coder-7B-LoRA/` - Checkpoints intermédiaires
- `complete_checkpoint/` - Checkpoint complet
- `final_model/` - Modèle final

## Télécharger les résultats

```bash
# Depuis votre machine locale
scp -P [SSH_PORT] -r root@[POD_IP]:/workspace/EndToEndLoRA/final_model ./
```