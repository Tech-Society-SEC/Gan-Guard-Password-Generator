import time
from zxcvbn import zxcvbn

# --- CONFIGURATION ---
SOURCE_FILE = 'rockyou.txt'
DEST_FILE = 'rockyou_common_subset.txt' # New output file
MIN_SCORE = 0  # We want the weak, common passwords
MAX_SCORE = 1  # (Scores 0 and 1)
# ---

def curate_passwords():
    print(f"Starting curation for 'common' passwords (score {MIN_SCORE}-{MAX_SCORE})...")
    start_time = time.time()
    count = 0
    with open(SOURCE_FILE, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(DEST_FILE, 'w', encoding='utf-8') as f_out:
        
        for i, line in enumerate(f_in):
            password = line.strip()
            if not password:
                continue

            # Check password strength
            try:
                results = zxcvbn(password)
                score = results['score']
            except Exception as e:
                continue # Skip weird passwords

            # Keep only the WEAK passwords (scores 0 and 1)
            if MIN_SCORE <= score <= MAX_SCORE:
                f_out.write(password + '\n')
                count += 1

            if (i + 1) % 1000000 == 0:
                print(f"Processed {i+1:,} lines, found {count:,} common passwords...")

    end_time = time.time()
    print(f"\n✅ Finished! Found {count:,} common passwords.")
    print(f"Saved to {DEST_FILE} in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    curate_passwords()