# Progetto di ARE2 - Sentiment Analysis

## Comanda del progetto

<h3>Subtasks B-C: Topic-Based Message Polarity Classification:  </h3>
Given a message and a topic, classify the message on   

  * **B) two-point scale:** positive or negative sentiment towards that topic
  * **C) five-point scale:** sentiment conveyed by that tweet towards the topic on a five-point scale.

## Struttura del progetto
Il progetto ha la seguente struttura:
  * **main.py** : main del progetto  
  * **utilities.py** : libreria che comprende una serie di funzioni utili utilizzate nel main
  * **tweets_subtask_BD** : all'interno della cartella ci sono i file di test, train, dev e devtest
  
  
## Struttura dei file
Ogni file presente nella cartella è così composto:  

  <table>
    <tr>
        <th><b>Tweet id</b></th>
        <th><b>Topic</b></th>
        <th><b>Tweet classification</b></th>
        <th><b>Tweet text</b></th>
    </tr>
    <tr>
        <td>522712800595300352</td>
        <td>aaron rodgers</td>
        <td>neutral</td>
        <td>I just cut a 25 second audio clip of Aaron Rodgers talking about Jordy Nelson's grandma's pies. Happy Thursday.</td>
    </tr>
    <tr>
        <td>523065089977757696</td>
        <td>aaron rodgers</td>
        <td>negative</td>
        <td>@Espngreeny I'm a Fins fan, it's Friday, and Aaron Rodgers is still giving me nightmares 5 days later. I wished it was a blowout.</td>
    </tr>
        <tr>
        <td>522477110049644545</td>
        <td>aaron rodgers</td>
        <td>positive</td>
        <td>Aaron Rodgers is really catching shit for the fake spike Sunday night.. Wtf. It worked like magic. People just wanna complain about the L.</td>
    </tr>
        <tr>
        <td>522551832476790784</td>
        <td>aaron rodgers</td>
        <td>neutral</td>
        <td>If you think  the Browns should or will trade Manziel you're an idiot. Aaron Rodgers sat behind Favre for multiple years.</td>
    </tr>
        <tr>
        <td>522887492333084674</td>
        <td>aaron rodgers</td>
        <td>neutral</td>
        <td>Green Bay Packers:  Five keys to defeating the Panthers in week seven: Aaron Rodgers On Sunday, ... http://t.co/anCHQjSLh9 #NFL #Packers</td>
    </tr>

  </table>
 
 ## Approccio alla risoluzione dei task
 * Preprocessing dei dati
 * Aggiunta di feature semantiche scaricate tramite Watson IBM (categorie, concetti) e inclusione di queste nei tweet
 * Utilizzo di TF-IDF
 * Creazione modello di Logistic Regression
 * Predizione
