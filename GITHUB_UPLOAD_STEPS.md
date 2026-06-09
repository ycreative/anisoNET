# GitHub Upload Steps

## Recommended First Repository

Create a private GitHub repository first:

- Repository name: `anisoNET`
- Visibility: Private
- Initialize with README: No
- Add `.gitignore`: No
- Add license: No, until the authors choose one

Private first is recommended because several upstream workflow scripts still contain local development defaults, although the current Figure 1-5 panel scripts have been parameterized for repository-root use.

## Upload By GitHub Web UI

1. Open the empty GitHub repository.
2. Click `uploading an existing file`.
3. Drag the contents of this folder, not the parent `github_staging` folder.
4. Commit with a message such as:

```text
Initial anisoNET reviewer reproducibility package
```

## Upload By Git Command Line

From this folder:

```bash
git init
git add .
git commit -m "Initial anisoNET reviewer reproducibility package"
git branch -M main
git remote add origin https://github.com/OWNER/anisoNET.git
git push -u origin main
```

Replace `OWNER` with the GitHub user or organization.

## After Upload

- Keep the repository private until final license, citation, author, and path-cleanup checks are complete.
- Add collaborators if coauthors need to inspect the package.
- Do not upload raw public datasets or full local analysis output trees to GitHub.
