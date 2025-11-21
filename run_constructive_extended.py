import os
import pandas as pd
import glob
import time
from datetime import datetime
from cvrp import *
from cvrp.solvers.constructive_heuristics.base_heuristics import ConstructiveHeuristicType
import concurrent.futures

INSTANCES_DIR = "vrp_instances" 
OUTPUT_DIR = "results_constructive_extended"

TIME_LIMIT_EXTENDED = 600    

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_experiment(instance, name, strategy, term_criteria):
    print(f"   -> Executando: {name}...")
    
    solver = CvrpTS(
        instance=instance,
        tenure=7,
        strategy=strategy,
        termination_criteria=term_criteria,
        debug_options=DebugOptions(verbose=False, log_history=True) 
    )
    
    start_t = time.time()
    solver.solve()
    end_t = time.time()
    
    initial_cost = solver.history[0][1] if solver.history else solver.best_solution.objfun_val
    final_cost = solver.best_solution.objfun_val
    improvement = ((initial_cost - final_cost) / initial_cost) * 100
    
  
    history_data = []
    for iter_num, best_val, curr_val, time_elapsed in solver.history:
        history_data.append({
            "Iteration": iter_num,
            "Time": time_elapsed,
            "BestCost": best_val,
            "CurrentCost": curr_val
        })
    
    inst_name_safe = instance.name.replace(' ', '_')
    exp_name_safe = name.replace(' ', '_')
    df_history = pd.DataFrame(history_data)
    df_history.to_csv(f"{OUTPUT_DIR}/hist_{inst_name_safe}_{exp_name_safe}.csv", index=False)
    
    return {
        "Instancia": instance.name,
        "Dimensao": instance.dimension,
        "Experimento": name,
        "Heuristica": strategy.constructive_heuristic.value,
        "EstrategiaBusca": strategy.search_strategy,
        "Intensificacao": strategy.enable_intensification,
        "Diversificacao": strategy.enable_diversification,
        "CustoInicial": initial_cost,
        "CustoFinal": final_cost,
        "MelhoriaPercentual": improvement,
        "TempoTotal": end_t - start_t,
        "Iteracoes": solver._iters
    }

def process_instance(instance_path):
    """Função wrapper para processar uma única instância inteira."""
    instance_filename = os.path.basename(instance_path)
    print(f"\n>>> Iniciando Thread para: {instance_filename}")
    
    try:
        instance = CvrpInstance.from_file(instance_path)
    except Exception as e:
        print(f"Erro ao ler {instance_filename}: {e}")
        return []

    results = []
    
    search_strategies = ['first', 'best']

  
    constructives = [
        ConstructiveHeuristicType.SAVINGS,
        ConstructiveHeuristicType.CHEAPEST_INSERTION,
        ConstructiveHeuristicType.ROUTE_FIRST_CLUSTER_SECOND
    ]
    
    for cons in constructives:
        for search_strat in search_strategies:
            strat = TSStrategy(
                constructive_heuristic=cons,
                search_strategy=search_strat, 
              
                neighborhoods={'relocate': True, 'swap': True, '2opt': True},
                enable_diversification=False, 
                enable_intensification=False
            )
            
          
          
            term = TerminationCriteria(
                max_time_secs=TIME_LIMIT_EXTENDED, 
                max_no_improvement=10000
            )
            
            res = run_experiment(instance, f"Req3_Ext_{cons.value}_{search_strat}", strat, term)
            res['DescricaoCenario'] = f"Construtiva Estendida ({search_strat})"
            results.append(res)

    print(f"<<< Finalizado: {instance_filename}")
    return results

def main():
    instance_files = sorted(glob.glob(os.path.join(INSTANCES_DIR, "*.vrp")))
    if not instance_files: 
        print("Nenhuma instância encontrada.")
        return

    all_results = []
    
  
    MAX_WORKERS = 8
    
    print(f"Iniciando processamento de construtivas (Extended) com {MAX_WORKERS} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_instance, f): f for f in instance_files}
        
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                data = future.result()
                all_results.extend(data)
            except Exception as exc:
                print(f'Gerou uma exceção: {exc}')

  
    if all_results:
        df_final = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"{OUTPUT_DIR}/final_results_constructive_extended_{timestamp}.csv"
        df_final.to_csv(csv_path, index=False)
        print(f"\nProcessamento Concluído! Arquivo salvo em: {csv_path}")
    else:
        print("Nenhum resultado gerado.")

if __name__ == "__main__":
    main()