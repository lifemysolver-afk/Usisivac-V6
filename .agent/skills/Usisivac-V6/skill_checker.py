import os
import sys

def check_skill(name, condition, error_msg):
    print(f"🔍 Provera skilla: {name}...", end=" ")
    if condition:
        print("✅")
        return True
    else:
        print(f"❌\n   Greška: {error_msg}")
        return False

def run_checks():
    project_root = os.getcwd()
    all_passed = True

    # 1. Config Management (Dynaconf)
    all_passed &= check_skill(
        "Advanced Config Management",
        os.path.exists(os.path.join(project_root, "settings.toml")),
        "Nedostaje 'settings.toml' fajl."
    )

    # 2. Structured Logging (Loguru)
    with open(os.path.join(project_root, "verify_ssot.py"), 'r') as f:
        content = f.read()
        all_passed &= check_skill(
            "Structured Logging",
            "loguru" in content and "logger.add" in content,
            "Loguru nije pravilno implementiran u 'verify_ssot.py'."
        )

    # 3. Data Validation (Cerberus)
    all_passed &= check_skill(
        "Data Validation",
        "cerberus" in content and "Validator" in content,
        "Cerberus validacija nedostaje u 'verify_ssot.py'."
    )

    if all_passed:
        print("\n🎉 Čestitamo! Projekat je u potpunosti usklađen sa GitHub Gem skillovima.")
    else:
        print("\n⚠️ Neki skillovi nisu pravilno implementirani. Pogledajte greške iznad.")
        sys.exit(1)

if __name__ == "__main__":
    run_checks()
