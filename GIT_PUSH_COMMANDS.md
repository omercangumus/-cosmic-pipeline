# Git Push Komutları

## Repository oluşturduktan sonra çalıştır:

```bash
# Remote'u kontrol et
git remote -v

# Eğer yanlış remote varsa sil ve tekrar ekle
git remote remove origin
git remote add origin https://github.com/omercangumus/cosmic-pipeline.git

# Develop branch'ini push et
git push -u origin develop

# Main branch'e geç ve merge et
git checkout -b main
git merge develop
git push -u origin main

# Tag ekle
git tag v0.1-day1-checkpoint
git tag v1.0-day2-complete
git push origin --tags
```

## Alternatif: Önce main branch'i push et

```bash
# Main branch oluştur ve push et
git checkout -b main
git push -u origin main

# Develop'a geri dön
git checkout develop
git push -u origin develop

# Feature branch'leri oluştur (opsiyonel)
git checkout -b feature/omer-day1-infra
git push -u origin feature/omer-day1-infra

git checkout -b feature/omer-day2-dashboard
git push -u origin feature/omer-day2-dashboard
```

## Tüm Değişiklikleri Görmek İçin:

```bash
git log --oneline --graph --all
```

## Durum Kontrolü:

```bash
git status
git branch -a
git remote -v
```
