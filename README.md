# SpaceshipTitanic
L'astronave Titanic è una nave passeggeri interstellare varata un mese fa. Con quasi 13.000 passeggeri a bordo, la nave ha intrapreso il suo viaggio trasportando emigranti dal nostro sistema solare a tre esopianeti.
Mentre doppiava Alpha Centauri in rotta verso la sua prima destinazione, 55 Cancri E, l’incauta astronave Titanic si scontrò con un'anomalia dello spaziotempo nascosta all'interno di una nuvola di polvere. Purtroppo, ha incontrato un destino simile al suo omonimo di 1000 anni prima. Anche se la nave è rimasta intatta, quasi la metà dei passeggeri è stata trasportata in una dimensione alternativa!
Per aiutare le squadre di soccorso a recuperare i passeggeri smarriti, devo prevedere quali passeggeri sono stati trasportati dall'anomalia utilizzando i dati recuperati dal sistema informatico danneggiato dell'astronave.
Dalla competizione kaggle Spaceship Titanic | Kaggle. ho 2 dataset: il training e lo score. Il training test è stato poi suddiviso in training e validation usando la funzione createDataPartition del pacchetto caret che crea una partizione stratificata per il target.  
Ogni osservazione (di tutti e tre i dataset) rappresenta un passeggero dell’astronave. 
In particolare, si conoscono:
•	caratteristiche specifiche del passeggero quali: id (PassengerId), età (Age), nome (name) ;
•	informazioni sul passeggero relative al viaggio: pianeta di partenza -tipicamente il pianeta di residenza-  (HomePlanet), ibernato -variabile binaria- (CryoSleep), destinazione (Destination), biglietto vip -variabile binaria- (VIP), numero della cabina (Cabin Num);
•	spese effettuate sulla navicella: RoomService, FoodCourt, ShoppingMall, Spa, VRDeck;
•	variabile risposta: transported (se è stato trasportato o no). Prendiamo come livello di riferimento “False” ovvero coloro che sopravvivono (non sono stati trasportati).
