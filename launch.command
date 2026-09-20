#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "======================================================"
echo "          Άρειος Πάγος — Αναζήτηση Νομολογίας         "
echo "======================================================"

# 1. Download pre-seeded modern corpus if no local database exists
if [ ! -f data/areios_pagos.db ] && [ ! -f data/areios_pagos_seed.db.gz ] && ! ls data/areios_pagos_seed.db.gz.part-* 1> /dev/null 2>&1; then
    echo "Λήψη προ-επεξεργασμένης βάσης δεδομένων (όλες οι αποφάσεις 2018-2026)..."
    mkdir -p data
    curl -L --progress-bar -o data/areios_pagos_seed.db.gz.part-aa "https://github.com/panlybero/areios-pagos-search/releases/download/v0.4.0/areios_pagos_seed.db.gz.part-aa"
    curl -L --progress-bar -o data/areios_pagos_seed.db.gz.part-ab "https://github.com/panlybero/areios-pagos-search/releases/download/v0.4.0/areios_pagos_seed.db.gz.part-ab"
fi

# 2. Join split database parts if present
if ls data/areios_pagos_seed.db.gz.part-* 1> /dev/null 2>&1 && [ ! -f data/areios_pagos_seed.db.gz ]; then
    echo "Συνένωση αρχείων βάσης δεδομένων..."
    cat data/areios_pagos_seed.db.gz.part-* > data/areios_pagos_seed.db.gz
fi

# 2. Ensure uv is available
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv &> /dev/null; then
    echo "Εγκατάσταση περιβάλλοντος εκτέλεσης (uv)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# 3. Run the application
echo "Εκκίνηση εφαρμογής και άνοιγμα στον περιηγητή..."
uv run apsearch launch "$@"
