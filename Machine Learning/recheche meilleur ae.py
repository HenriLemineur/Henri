import re

def parse_training_results(file_path):
    results = []
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Rechercher toutes les sections de résultats
        sections = re.findall(r"Mode: (.+?), Learning Rate: (.+?)\n\nTraining History:\n(.+?)\n\n", content, re.DOTALL)
        
        for section in sections:
            mode = section[0].strip()
            learning_rate = float(section[1].strip())
            history_str = section[2].strip()
            
            # Extraire les valeurs de la perte
            loss_match = re.search(r"loss: \[(.+?)\]", history_str)
            val_loss_match = re.search(r"val_loss: \[(.+?)\]", history_str)
            
            if loss_match and val_loss_match:
                loss_values = list(map(float, loss_match.group(1).split(',')))
                val_loss_values = list(map(float, val_loss_match.group(1).split(',')))
                
                # Calculer la perte finale
                final_loss = loss_values[-1] if loss_values else float('inf')
                final_val_loss = val_loss_values[-1] if val_loss_values else float('inf')
                
                results.append({
                    'mode': mode,
                    'learning_rate': learning_rate,
                    'final_loss': final_loss,
                    'final_val_loss': final_val_loss
                })
    
    return results

def find_best_parameters(results):
    # Trouver le meilleur résultat basé sur la perte de validation finale
    best_result = min(results, key=lambda x: x['final_val_loss'])
    return best_result

def main():
    file_path = 'training_results_5.txt'
    
    # Parser les résultats
    results = parse_training_results(file_path)
    
    # Trouver les meilleurs paramètres
    best_result = find_best_parameters(results)
    
    print(f"Meilleurs paramètres trouvés :")
    print(f"Mode: {best_result['mode']}")
    print(f"Learning Rate: {best_result['learning_rate']}")
    print(f"Perte finale (entraînement): {best_result['final_loss']}")
    print(f"Perte finale (validation): {best_result['final_val_loss']}")

if __name__ == '__main__':
    main()
