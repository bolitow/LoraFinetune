# Guide de déploiement RunPod pour Qwen Coder v2.5 7B LoRA

## Prérequis

- Git LFS installé localement (le projet utilise des fichiers volumineux)
- Token HuggingFace avec accès au modèle Qwen2.5-Coder-7B
- Compte RunPod avec crédits

## 1. Préparer votre projet localement

```bash
# S'assurer que Git LFS est installé
git lfs install

# Commit vos changements
git add .
git commit -m "Update for Qwen Coder v2.5 7B training"
git push
```

## 2. Créer un pod RunPod

1. Allez sur [RunPod.io](https://runpod.io)
2. Créez un nouveau pod avec :
   - **GPU**: RTX 6000 Ada (48GB VRAM recommandé) ou RTX 4090/A100 (24GB minimum)
   - **Template**: PyTorch 2.1 + CUDA 12.1
   - **Disk**: 100GB minimum (pour accommoder le modèle et les données LFS)
   - **Options**: SSH activé
   
   Note: Avec RTX 6000 Ada (48GB), le training est optimisé pour des performances maximales

## 3. Se connecter au pod

```bash
ssh root@[YOUR_POD_IP] -p [YOUR_SSH_PORT]
```

## 4. Cloner votre projet

```bash
cd /workspace
# Le script runpod_deploy.sh va automatiquement cloner le repo avec Git LFS
# Si vous préférez le faire manuellement :
# git clone https://github.com/bolitow/LoraFinetune.git EndToEndLoRA
# cd EndToEndLoRA
# git lfs pull
```

## 5. Configurer HuggingFace

```bash
# Obtenir un token depuis https://huggingface.co/settings/tokens
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
```

## 6. Configuration pour publication automatique (optionnel)

```bash
# Pour publier automatiquement le modèle sur HuggingFace
export HF_USERNAME="votre-username-huggingface"
```

## 7. Lancer le script de déploiement

```bash
./runpod_deploy.sh
```

Ce script va automatiquement :
- Installer Git LFS sur le pod
- Cloner le repository avec les fichiers LFS
- Installer uv et toutes les dépendances
- Configurer l'environnement d'entraînement

## 8. Lancer l'entraînement

```bash
uv run train.py
```

## 9. Surveiller l'entraînement

Dans un autre terminal SSH :
```bash
watch -n 1 nvidia-smi
```

## Paramètres d'entraînement optimisés pour RTX 6000 Ada

Le script détecte automatiquement votre GPU et optimise les paramètres :

### Configuration RTX 6000 Ada (48GB) :
- **Batch size**: 8 par GPU
- **Gradient accumulation**: 2 steps (batch effectif = 16)
- **LoRA rank**: 512 avec RSLoRA
- **Learning rate**: 5e-5 avec cosine decay
- **Precision**: BF16 + TF32 + Flash Attention 2
- **Optimisations**: Sequence packing, paged AdamW
- **Max sequence length**: 2048 tokens

### Configuration RTX 4090/A100 (24GB) :
- **Batch size**: 1-2 par GPU
- **Gradient accumulation**: 4-8 steps
- **LoRA rank**: 256
- **Mixed precision**: FP16

## Après l'entraînement

Le modèle final fusionné sera sauvegardé dans :
- `final_merged_model/` - Modèle complet avec poids LoRA fusionnés

Si HF_USERNAME est configuré :
- Publication automatique sur HuggingFace (repository privé)
- Nom du repo : `qwen2.5-coder-7b-lora-YYYYMMDD-HHMMSS`

## Télécharger les résultats

```bash
# Depuis votre machine locale
scp -P [SSH_PORT] -r root@[POD_IP]:/workspace/EndToEndLoRA/final_merged_model ./
```

## Statistiques de performance

Le script affiche automatiquement :
- Temps total d'entraînement
- Loss final
- Métriques de performance
- Lien HuggingFace si publié