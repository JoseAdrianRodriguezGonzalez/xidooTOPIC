#git tag -d v0.1.0-rc5
#git push origin --delete v0.1.0-rc5
git add .
git commit -m "Changing some of the namespace"
git push
git tag v0.1.0-rc5
git push origin v0.1.0-rc5
