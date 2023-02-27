# SpaceshipTitanic

L'astronave Titanic è una nave passeggeri interstellare varata un mese fa. Con quasi 13.000 passeggeri a bordo, la nave ha intrapreso il suo viaggio trasportando emigranti dal nostro sistema solare a tre esopianeti.
Mentre doppiava Alpha Centauri in rotta verso la sua prima destinazione, 55 Cancri E, l’incauta astronave Titanic si scontrò con un'anomalia dello spaziotempo nascosta all'interno di una nuvola di polvere. Purtroppo, ha incontrato un destino simile al suo omonimo di 1000 anni prima. Anche se la nave è rimasta intatta, quasi la metà dei passeggeri è stata trasportata in una dimensione alternativa!
Per aiutare le squadre di soccorso a recuperare i passeggeri smarriti, devo prevedere quali passeggeri sono stati trasportati dall'anomalia utilizzando i dati recuperati dal sistema informatico danneggiato dell'astronave.
Dalla competizione kaggle [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic). 

Ho 2 dataset: il training e lo score. Il training test è stato poi suddiviso in training e validation usando la funzione createDataPartition del pacchetto caret che crea una partizione stratificata per il target.  
Ogni osservazione (di tutti e tre i dataset) rappresenta un passeggero dell’astronave. 

In particolare, si conoscono:

- caratteristiche specifiche del passeggero quali: id (PassengerId), età (Age), nome (name) ;

- informazioni sul passeggero relative al viaggio: pianeta di partenza -tipicamente il pianeta di residenza-  (HomePlanet), ibernato -variabile binaria- (CryoSleep), destinazione (Destination), biglietto vip -variabile binaria- (VIP), numero della cabina (Cabin Num);

- spese effettuate sulla navicella: RoomService, FoodCourt, ShoppingMall, Spa, VRDeck;

- variabile risposta: transported (se è stato trasportato o no). Prendiamo come livello di riferimento “False” ovvero coloro che sopravvivono (non sono stati trasportati).

Distribuzione dei soggetti rispetto alla variabile target: 

![Picture1](https://user-images.githubusercontent.com/85078090/221669399-4cefcf04-bf5a-4488-996a-59c8c5494efc.png)

in questo caso non ho un dataset sbilanciato e inoltre posso assumere che le proporzioni del dataset corrispondono alle proporzioni reali della popolazione in quanto sto affrontando un problema che è stato inventato. Nel caso in cui avessi avuto proporzioni del dataset diverse da quelle reali avrei dovuto correggere le probabilità a posteriori trovate in output dai modelli classificativi con le probabilità a priori reali dei due eventi.
Prima di iniziare a implementare algoritmi classificativi vado ad analizzare il dataset con cui sto lavorando per capire come sono i dati e come sono distribuiti. 

![image](https://user-images.githubusercontent.com/85078090/221669622-b8a22bda-83e8-45af-af4a-bc49a427f007.png)

![image](https://user-images.githubusercontent.com/85078090/221669643-53642e66-4d60-4701-bcf7-9df3e1193969.png)


Importante è capire se vi sono problemi di separation (ovvero capire se ho una variabile che identifica perfettamente il target), e in caso affermativo eliminarla. Nel mio caso non ce ne sono. Si può notare che se un paziente è sato ibernato allora ha molta più probabilità di essere stato trasportato, coerentemente con il fatto che i pazienti a riposo non hanno potuto lasciare le loro cabine.
Degni di nota sono anche i passeggeri che hanno acquistato il biglietto VIP, nonostante essi siano solo il 2,2% della popolazione più della metà di quest’ultimi non viene trasportata.

Vado ad apportare qualche modifica alle variabili del dataset per renderle più consone alla classificazione:
* il numero di cabina indica il posto in cui soggiorna il passeggero ed è formato da una tripletta contenetene le informazioni riguardante il ponte, il numero e il lato. Suddividiamo dunque la variabile cabina in tre variabili diverse e osserviamo come si distribuiscono il numero dei trasportati in funzione di queste tre nuove variabili.

![image](https://user-images.githubusercontent.com/85078090/221671687-c2ab46c3-7085-46a8-8e6c-9a4312b1b926.png)

![image](https://user-images.githubusercontent.com/85078090/221671715-569984f1-dfa0-44f4-8438-656be00bfe18.png)

* Passenger id è un id univoco nella forma gggg_pp dove gggg rappresenta il gruppo con cui il passeggero sta viaggiando mentre pp è il suo numero all’interno del gruppo. Abbiamo quindi ricavato l’idGroup 
e per ogni gruppo abbiamo calcolato il numero di membri.
Numero di trasportati e non trasportati in funzione del numero di componenti del gruppo. 

![image](https://user-images.githubusercontent.com/85078090/221671735-1e39ca41-292e-4526-9f3b-77f6f5b14457.png)

La maggior parte degli algoritmi non è in grado di gestire i dati mancanti di conseguenza, dopo averli analizzati, vado ad imputarli.
Ho circa il 2% di missing sul totale dei passeggeri e sono equamente distribuiti fra le variabili. 

![image](https://user-images.githubusercontent.com/85078090/221672009-b2621e40-71e6-4e50-a977-6e226b580245.png)

Per imputarli ho utilizzato la funzione mice dell’omonimo pacchetto che implementa il metodo fcs con l’albero come modello. 
Ciò significa che ho fatto imputazione utilizzando l’albero ovvero il mio modello ha preso come variabile target una covariata contenente dati mancanti e ha costruito un albero con le restanti facendo previsione o classificazione (sulla base del tipo di covariata presa come target) e dunque sostituendo i valori previsti/classificati con i valori mancanti per quella covariata. Itera su tutte le variabili contenenti valori mancanti.


Prima di iniziare a costruire un modello è importante lavorare sulla complessità dei dati. Il modo più semplice e veloce per ridurre la dimensionalità è quello di effettuare model selection.

Boruta in pochi passaggi:
1. crea una copia del dataset iniziale
2. va a permutare le osservazioni di un'unica variabile (chiamata Shadow feature) che sarà quella che si andrà a verificare se importante
3. vado poi a costruire un albero come modello su entrambi i dataset e vado a comparare le 2 accuratezze trovate. Nel caso in cui l’accuratezza non diminuisce significa che quella variabile non dava contributo nella classificazione e di conseguenza può essere rimossa.
4. Ripeto per ogni variabile 

![image](https://user-images.githubusercontent.com/85078090/221672157-1b9edc52-9100-45fb-ae72-9510ca853d20.png)

l’unica variabile che non è stata considerata importante è VIP.
Creo un nuovo dataset senza la variabile VIP da dare in pasto a quegli algoritmi classificativi che richiedono model selection.

Dopo aver creato il dataset senza dati mancanti e un dataset con le variabili importanti posso iniziare a implementare qualche modello classificativo.

Per tunare i parametri uso come metrica la sensitivity perché preferisco capire quali sono i passeggeri che non sono stati trasportati in un'altra dimensione per andare a cercarli ed è meglio andare a cercare un passeggero che è stato trasportato nell’altra dimensione (erroneamente) piuttosto che non cercare un passeggero che non è stato trasportato ma che il mio algoritmo mi classifica come trasportato. Quindi preferisco avere più falsi trasportati che non falsi non trasportati.
Tutti i modelli sono stati fittati con la funzione train di caret ad eccezione dello stacking.
Riporto solo i risultati ottenuti dai vari modelli che ho implementato.

# **Logistico**

Preprocess: variabili non collineari, near zero variance e model selection

![image](https://user-images.githubusercontent.com/85078090/221672265-bc16470f-705a-4b2d-9966-f83e27eb59e9.png)

![image](https://user-images.githubusercontent.com/85078090/221672283-4c254496-0fbe-42d6-accd-52b51671500d.png)

![image](https://user-images.githubusercontent.com/85078090/221672293-6ab98fc1-c290-49e0-920a-01a8263895a8.png)

# **Knn**  

Preprocess: variabili non collineari, near zero variance, model selection, variabili standardizzate e variabili fattoriali rese continue. 
Come parametro di tuning abbiamo K = 23 (il numro di vicini), tunato dal modello. 

![image](https://user-images.githubusercontent.com/85078090/221676347-f1d23a36-c8b0-4162-b652-b17986997c6c.png)

![image](https://user-images.githubusercontent.com/85078090/221676535-ad96147d-d68b-40c2-8e05-e24d4758d172.png)

![image](https://user-images.githubusercontent.com/85078090/221676590-8e09f990-fcbc-4228-b87a-f17baad4c481.png)

# **Gradient boosting**

Parametro di tuning: numero di alberi (500) 

![image](https://user-images.githubusercontent.com/85078090/221676835-8497a7f7-bbf3-4287-a123-189313685b4e.png)

![image](https://user-images.githubusercontent.com/85078090/221676854-57316258-fe82-42b3-b678-82e1477a8c3f.png)

![image](https://user-images.githubusercontent.com/85078090/221676891-5fd5db77-1629-4894-9ad4-0e799f03e889.png)

# **Random forest**

Parametro di tuning: numero di features considerato ad ogni split (7)

![image](https://user-images.githubusercontent.com/85078090/221676998-fa281f0d-d767-4852-8359-43664d86bcd5.png)

![image](https://user-images.githubusercontent.com/85078090/221677018-c2f805a5-0bb6-467b-bc17-11d32daaedd0.png)

![image](https://user-images.githubusercontent.com/85078090/221677053-163fbc2f-aa26-4361-90c9-d5d0213b931b.png)

# **Rete neurale**

Parametri da tunare: numero di strati (5) e decay (0.1)
Preprocess: variabili standardizzate, non collineari e near zero variance

![image](https://user-images.githubusercontent.com/85078090/221677161-27a38bde-8c43-47b6-b069-09ce062edfa6.png)

![image](https://user-images.githubusercontent.com/85078090/221677174-9ca8432d-ffe0-46e3-a095-cf68647e7f2d.png)

![image](https://user-images.githubusercontent.com/85078090/221677193-ac81278e-1b9d-4d3d-8884-c7c1f38e9f94.png)

# **Stacking – glm**

Modelli usati: logistico, albero, knn, sda (shrinkage discriminant analysis), lasso, pls, naive bayes, gradient boosting, random forest, reti neurali, bagging. 
Metamodello logistico di coefficienti: 	

![image](https://user-images.githubusercontent.com/85078090/221677386-316c1772-7791-43f6-aef7-d6d473512e1a.png)

![image](https://user-images.githubusercontent.com/85078090/221677415-d9617235-3321-4097-9b72-a02212595eb7.png)

Per confrontare i vari modelli non possiamo basarci su quello che va a massimizzare l’accuracy perché l’accuratezza dipende dalla soglia con cui decidiamo di classificare e di conseguenza non è un metodo robusto. Bisogna confrontare i modelli per tutte le possibili soglie. 
Grafico delle curve ROC:

![image](https://user-images.githubusercontent.com/85078090/221677480-398c66ca-fc4c-4680-a540-06166d026d58.png)

Dal grafico delle curve ROC si può notare che il knn e la random forest sono i due metodi che forniscono i modelli classificativi peggiori. Mentre le curve migliori son quelle del gradient boosting, rete neurali e stacking. Queste tre curve però si sovrappongono ed è difficile valutare quale sia effettivamente la migliore. Ha senso allora andare a studiare le curve lift di questi tre modelli. Per ogni percentuale della popolazione (asse delle x) mi restituisce la percentuale cumulata di non trasportati catturata (ovvero i classificati in modo corretto).

Gradient boosting: 

![image](https://user-images.githubusercontent.com/85078090/221677907-8028672b-b0aa-499c-bd90-e261300f0b05.png)

Neural network: 

![image](https://user-images.githubusercontent.com/85078090/221677954-4923052f-91ef-4054-9e41-41b7f70866a6.png)

Staking – glm:

![image](https://user-images.githubusercontent.com/85078090/221677999-2d051f89-e144-44f0-b63e-d8886586f8bc.png)

Ho selezionato come modello lo stacking essendo quello in grado di catturare il maggior numero di non trasportati (60,5%) considerando solo il 30% della popolazione. 






