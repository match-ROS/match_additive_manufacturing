# GitHub Pages deployment notes

## Option 1: user or organization page
1. Create a repository named `<username>.github.io`.
2. Upload all files from this folder to the repository root.
3. Commit and push.
4. Your page will be available at `https://<username>.github.io/`.

## Option 2: project page
1. Put these files into a `docs/` folder inside your project repository.
2. In GitHub, open **Settings → Pages**.
3. Select **Deploy from a branch**.
4. Choose your main branch and the `/docs` folder.
5. Save.
6. Your page will be available at `https://<username>.github.io/<repo-name>/`.

## Recommended small edits
- Replace the placeholder citation with the final journal metadata.
- Replace page screenshots with cropped figures exported from the paper for cleaner visuals.
- Add a teaser video or GIF if you have one.
- Add links to supplementary material, datasets, and BibTeX.
