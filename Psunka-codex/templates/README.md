# UI templates

Binary screenshots are intentionally not committed to the repository.

Place the real PNG templates captured from the target Toloka UI in this directory before enabling mouse automation:

- `play.png`
- `pause.png`
- `loading.png`
- `category_panel.png`
- `send_button.png`
- `text_field.png`
- `category_1_checked.png`
- `category_1_unchecked.png`
- `category_2_checked.png`
- `category_2_unchecked.png`
- `category_3_checked.png`
- `category_3_unchecked.png`
- `category_4_checked.png`
- `category_4_unchecked.png`

The code checks these files at runtime and reports missing templates instead of clicking arbitrary coordinates.
