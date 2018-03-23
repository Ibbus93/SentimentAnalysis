# <a href="https://github.com/Ibbus93/">Federico Ibba</a>, <a href="https://github.com/ferruvich">Daniele Stefano Ferru</a>, <a href="http://people.unica.it/diegoreforgiato/">Diego Reforgiato
 
# Supervised Topic-Based Message Polarity Classification using Cognitive Computing

This project is part of <a href="http://alt.qcri.org/semeval2017/task4/">Semeval 2017 Task 4</a>  


## Project tasks

<h3>Subtasks B-C: Topic-Based Message Polarity Classification:  </h3>
Given a message and a topic, classify the message on   

  * **B) two-point scale:** positive or negative sentiment towards that topic
  * **C) five-point scale:** sentiment conveyed by that tweet towards the topic on a five-point scale.

## Project structure
The project is built as follow:
  * **are_project_B.py** : execution file task B
  * **are_project_C.py** : execution file task C
  * **utilities.py** : personal library with various useful functions used by tasks
  * **train_BD.tsv** : train dataset task B
  * **train_CE.tsv** : train dataset task C
  * **test_BD.tsv** : test dataset task B
  * **test_CE.tsv** : test dataset task C
  
## Dataset structure
The following table explains how the dataset is composed:  

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
 
## Tasks resolution approach
<ol>
 <li>Data Preprocessing</li>     
 <li>Every record has been associated with categories and concepts taken by IBM Watson</li>
 <li>Various classifiers has been trained to obtain the best obtainable scores requested by the challenge.</li>
 <li>Best results has been taken</li> 
</ol>

## Results of the studied case
The results of this research has been written into a paper proposed to <a href="http://www.maurodragoni.com/research/opinionmining/events/">Workshop on Sentic Computing, Sentiment Analysis, Opinion Mining, and Emotion Detection</a>.
