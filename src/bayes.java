import java.lang.Math;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import weka.core.*;

public class bayes{
	private int LABEL = 2;
	private int PAL = 1;
	//for grams
	private ArrayList<String> grams = new ArrayList<String>();
	private ArrayList<Integer> abNum = new ArrayList<Integer>();
	private ArrayList<Integer> nmNum = new ArrayList<Integer>();
	private ArrayList<Float> prob = new ArrayList<Float>();
	private ArrayList<Float> bayesian = new ArrayList<Float>();
	private Instances trainset;
	private Instances testset;
	private int ab=0;
	private int nm=0;
	private int timesOfAb=0;
	private int timesOfNm=0;
	// for test data
	private float[] TF = new float[4];
	private float shrshd = (float) 0.55;
	private int abTest=0;
	private int nmTest=0;
	private int abCorr=0;
	private int nmCorr=0;
	// for 
	//
	private ArrayList<String>kwInNorm = new ArrayList<String>();
	private ArrayList<String>kwInAnom = new ArrayList<String>();
	private ArrayList<Integer>kwTimesInNorm = new ArrayList<Integer>();
	private ArrayList<Integer>kwTimesInAnom = new ArrayList<Integer>();
	private ArrayList<String>valueInNorm = new ArrayList<String>();
	private ArrayList<String>valueInAnom = new ArrayList<String>();
	private ArrayList<Integer>valueTimesInNorm = new ArrayList<Integer>();
	private ArrayList<Integer>valueTimesInAnom = new ArrayList<Integer>();
	//
	private ArrayList<String>kwInTrain= new ArrayList<String>();
	private ArrayList<Integer>features=new ArrayList<Integer>();
	// for combined
	private ArrayList<Integer> combined_result=new ArrayList<Integer>();
	private ArrayList<Integer> label_real=new ArrayList<Integer>();
	private ArrayList<Integer> label_oneClass=new ArrayList<Integer>();
	private ArrayList<Integer> label_twoClass=new ArrayList<Integer>();
	//
	private int[] getFeature(String kwValue){
		int[]feature =new int[4];//length,#ofWords,#ofSymbles,#ofCn
		String[] temp=kwValue.split(" ");
		String str=kwValue.replaceAll("[\u4E00-\u9FA5]","");
		feature[3]=kwValue.length()-str.length();
		//System.out.println(feature[3]+" ");
		feature[0]=kwValue.length();
		feature[1]=temp.length;
		kwValue=kwValue.replaceAll("[A-Za-z0-9\\s]","");
		feature[2]=kwValue.length()-feature[3];
		return feature;
	}
	private void TrainFeature(Instances Inst) {
		String pLoad;
		String kw;
		String isNorm;
		for(int i=0;i<Inst.numInstances();i++){
			isNorm=Inst.instance(i).toString(2);
			if(isNorm.equals("norm")){
				//System.out.println(isNorm);
				pLoad=(Inst.instance(i).toString(1));
				String[] temp=pLoad.split("=");
				int []feature=getFeature(temp[1]);
				//System.out.println(feature.length);
				if (!kwInTrain.contains(temp[0])){
					kwInTrain.add(temp[0]);
					for(int k=0;k<feature.length;k++){
						features.add(4*kwInTrain.indexOf(temp[0])+k,feature[k]);
					}
				}
				int idxOfkw=kwInTrain.indexOf(temp[0]);
				for(int j=0;j<feature.length;j++){
					if(features.get(4*idxOfkw+j)<feature[j]){
						features.set(4*idxOfkw+j, feature[j]);
					}
				}
			}
		}
		System.out.println(features.size());
		System.out.println(features);
	}
	private void TestFeature(Instances Inst,String path) throws IOException{
		FileWriter writer;
		writer=new FileWriter(path+"resultByFeatures.txt");
		ArrayList<String> packScore=new ArrayList<String>();
		String isNorm;
		int[] amount=new int[4];//totalPackAmount,totalAnomPackAmount,checkedAnomPackAmount,CheckedNormPackAmount
		int pNum=0;
		int idx=0;
		String pLoad;
		String kw;
		String value;
		for(int i=0;i<Inst.numInstances();i++){
			pNum=Integer.parseInt(Inst.instance(i).toString(0));
			isNorm=Inst.instance(i).toString(2);
			if(isNorm.equals("anom")) label_real.add(1);
			else label_real.add(0);
			amount[0]++;
			if(isNorm.equals("anom")) amount[1]++;
			while(i<Inst.numInstances()&&(idx=Integer.parseInt(Inst.instance(i).toString(0)))==pNum){
				pLoad=Inst.instance(i).toString(1);
				String[]temp=pLoad.split("=");
				kw=temp[0];
				value=temp[1];
				int nCn=0;
				String str=temp[1].replaceAll("[\u4E00-\u9FA5]","");
				nCn=temp[1].length()-str.length();
				//if(nCn>0)packScore.add("CN");
				if(kwInTrain.contains(kw)){
					packScore.add("kwIn");
					int[]feature=getFeature(value);
					for(int j=0;j<feature.length;j++){
						if(feature[j]>features.get(kwInTrain.indexOf(kw)*4+j)){
							packScore.add("vOut");
						}
					}
				}
				else{
					packScore.add("kwOut");
				}
				i++;
			}
			i--;
			if(packScore.contains("kwOut")||packScore.contains("vOut")||packScore.contains("CN")){
				packScore.add(0,"isAnom");
				label_oneClass.add(1);
				if(isNorm.equals("anom")) {amount[2]++; packScore.add(0,"Checked ");}
				else {packScore.add(0, "unChecked"); amount[3]++;}
				
			}
			else{
				//amount[3]++;
				label_oneClass.add(0);
				packScore.add(0,"IsNorm");
				if(isNorm.equals("norm")) packScore.add(0,"Checked ");
				else packScore.add(0, "unChecked");
			}
			for(int k=0;k<packScore.size();k++){
				writer.append(packScore.get(k));
			}
			
			writer.append("\n");
			packScore.clear();
		}
		float tp=(float)amount[2]/amount[1];
		float fp=(float)amount[3]/(amount[0]-amount[1]);
		writer.append("TP:"+tp+"\n");
		writer.append("FP:"+fp+"\n");
		writer.flush();
		writer.close();
		for(int n=0;n<amount.length;n++){
			System.out.println(amount[n]);
		}
		System.out.println(tp);
		System.out.println(fp);

	}
	//
	private void getFirstKwNorm(Instances Inst, String path)throws IOException{
		String kw;
		String w1 ="";
		String w2="";
		String idx;
		String idx1;
		String idx2;
		int i;
		int packet=1;
		int index=0;
		int index1=0;
		int index2=0;
		for(i=0;i<Inst.numInstances();i++){
			System.out.println("I:"+i);
			idx=Inst.instance(i).toString(0);
			index=Integer.parseInt(idx);
			if(index==packet&&packet<35999){
				kw=Inst.instance(i).toString(1);
				if(kw.charAt(0)=='\''){kw.replaceAll("\'", "");}
				if(kw.contains("=")){
					String[] temp = kw.split("=");
					w1=temp[0];
					w2 = temp[1];
				}
				else w1=kw;
				if(!kwInNorm.contains(w1)){
					kwInNorm.add(w1);
					kwTimesInNorm.add(1);
				}
				else{
					int pos=kwInNorm.indexOf(w1);
					kwTimesInNorm.set(pos, valueTimesInNorm.get(pos)+1);
				}
				if(!valueInNorm.contains(w2)){
					valueInNorm.add(w2);
					valueTimesInNorm.add(1);
				}
				else{
					int pos=valueInNorm.indexOf(w2);
					valueTimesInNorm.set(pos, valueTimesInNorm.get(pos)+1);
				}
				int j=i+1;
				idx1=Inst.instance(i).toString(0);
				index1=Integer.parseInt(idx1);
				idx2=Inst.instance(j).toString(0);
				index2=Integer.parseInt(idx2);
				while(index1==index2&&j<Inst.numInstances())
				{
					j++;
					idx2=Inst.instance(j).toString(0);
					index2=Integer.parseInt(idx2);
				}
				packet=index2;
			}
			System.out.println(packet);
		}
		System.out.println(packet);
		FileWriter writer1 = new FileWriter(path+"startingKwNorm.txt");
		FileWriter writer2 = new FileWriter(path+"startingValueNorm.txt");
		writer1.append("StartingKW"+"\t"+"times"+"\n");
		writer2.append("values"+"\t"+"times"+"\n");
		for(i=0;i<kwInNorm.size();i++){
			writer1.append(kwInNorm.get(i)+"\t"+kwTimesInNorm.get(i)+"\n");
		}
		for(i=0;i<valueInNorm.size();i++){
			writer2.append(valueInNorm.get(i)+"\t"+valueInNorm.get(i)+"\n");
		}
		writer1.flush();
		writer2.flush();
		writer1.close();
		writer2.close();
	}
	private void getFirstKwAnom(Instances Inst, String path)throws IOException{
		String kw;
		String w1 ="";
		String w2="";
		String idx;
		String idx1;
		String idx2;
		int i;
		int packet=0;
		int index=0;
		int index1=0;
		int index2=0;
		for(i=0;i<Inst.numInstances();i++){
			System.out.println("I:"+i);
			idx=Inst.instance(i).toString(0);
			index=Integer.parseInt(idx);
			if(index==packet&&packet<25064){
				kw=Inst.instance(i).toString(1);
				if(kw.charAt(0)=='\''){kw.replaceAll("\'", "");}
				String[] temp = kw.split("=");
				w1=temp[0];
				w2 = temp[1];
				if(!kwInAnom.contains(w1)){
					kwInAnom.add(w1);
					kwTimesInAnom.add(1);
				}
				else{
					int pos=kwInAnom.indexOf(w1);
					kwTimesInAnom.set(pos, valueTimesInAnom.get(pos)+1);
				}
				if(!valueInAnom.contains(w2)){
					valueInAnom.add(w2);
					valueTimesInAnom.add(1);
				}
				else{
					int pos=valueInAnom.indexOf(w2);
					valueTimesInAnom.set(pos, valueTimesInAnom.get(pos)+1);
				}
				int j=i+1;
				idx1=Inst.instance(i).toString(0);
				index1=Integer.parseInt(idx1);
				idx2=Inst.instance(j).toString(0);
				index2=Integer.parseInt(idx2);
				while(index1==index2&&j<Inst.numInstances())
				{
					j++;
					idx2=Inst.instance(j).toString(0);
					index2=Integer.parseInt(idx2);
				}
				packet=index2;
			}
			//System.out.println(packet);
		}
		System.out.println(packet);
		FileWriter writer1 = new FileWriter(path+"startingKwAnom.txt");
		FileWriter writer2 = new FileWriter(path+"startingValueAnom.txt");
		writer1.append("StartingKW"+"\t"+"times"+"\n");
		writer2.append("values"+"\t"+"times"+"\n");
		for(i=0;i<kwInAnom.size();i++){
			writer1.append(kwInAnom.get(i)+"\t"+kwTimesInAnom.get(i)+"\n");
		}
		for(i=0;i<valueInAnom.size();i++){
			writer2.append(valueInAnom.get(i)+"\t"+valueTimesInAnom.get(i)+"\n");
		}
		writer1.flush();
		writer2.flush();
		writer1.close();
		writer2.close();
	}
	private void getAnomKW(Instances Inst, String path) throws IOException{
		int i=0;
		int idx=0;
		String kw;
		String w1 = null;
		String w2 = null;
		for(i=0;i<Inst.numInstances();i++){
			System.out.println(i);
			w1=null;
			w2=null;
			kw=Inst.instance(i).toString(1);
			if(kw.charAt(0)=='\''){kw.replaceAll("\'", "");}
			if(kw.contains("=")){
				String[] temp = kw.split("=");
				w1=temp[0];
				w2 = temp[1];
			}
			if(!kw.contains("=")){
				w1=w2=kw;
			}
			if(!valueInAnom.contains(w2)){
				valueInAnom.add(w2);
				valueTimesInAnom.add(1);
			}
			else{
				idx=valueInAnom.indexOf(w2);
				valueTimesInAnom.set(idx, valueTimesInAnom.get(idx)+1);
			}
			if(!kwInAnom.contains(w1)){
				kwInAnom.add(w1);
				kwTimesInAnom.add(1);
			}
			else{
				idx=kwInAnom.indexOf(w1);
				kwTimesInAnom.set(idx, kwTimesInAnom.get(idx)+1);
			}			
		}
		FileWriter writer1;
		FileWriter writer2;
		writer1 = new FileWriter(path+"keywordsInAnom.txt");
		writer2 = new FileWriter(path+"valuesInAnom.txt");
		writer1.append("Abnormal keywords"+"\t"+"times"+"\t"+"\n");
		writer2.append("values"+"\t"+"times"+"\n");
		for(i=0;i<kwInAnom.size();i++){
			writer1.append(kwInAnom.get(i)+"\t"+kwTimesInAnom.get(i)+"\n");
			//System.out.println(kwInAnom.get(i)+" "+kwTimesInAnom.get(i));
		}
		for(i=0;i<valueInAnom.size();i++){
			writer2.append(valueInAnom.get(i)+"\t"+valueTimesInAnom.get(i)+"\n");
		}
		writer1.flush();
		writer1.close();
		writer2.flush();
		writer2.close();
	}
	private void getNormKW(Instances Inst, String path) throws IOException{
		int i=0;
		int idx=0;
		String kw;
		String w1 = null;
		String w2 = null;
		for(i=0;i<Inst.numInstances();i++){
			w1=w2="";
			kw=Inst.instance(i).toString(1);
			if(kw.charAt(0)=='\''){kw.replaceAll("\'", "");}
			if(kw.contains("=")){
				String[] temp = kw.split("=");
				w1=temp[0];
				for(int j=1;j<temp.length;j++){
					w2 += temp[j];
				}
			}
			if(!kwInNorm.contains(w1)){
				kwInNorm.add(w1);
				kwTimesInNorm.add(1);
			}
			else{
				idx=kwInNorm.indexOf(w1);
				kwTimesInNorm.set(idx, kwTimesInNorm.get(idx)+1);
			}
			if(!valueInNorm.contains(w2)){
				valueInNorm.add(w2);
				valueTimesInNorm.add(1);
			}
			else{
				idx=valueInNorm.indexOf(w2);
				valueTimesInNorm.set(idx, valueTimesInNorm.get(idx)+1);
			}			
		}
		FileWriter writer1;
		FileWriter writer2;
		writer1 = new FileWriter(path+"keywordsInNorm.txt");
		writer2 = new FileWriter(path+"valuesInNorm.txt");
		writer1.append("Normal keywords"+"\t"+"times"+"\n");
		writer2.append("values"+"\t"+"times"+"\n");
		for(i=0;i<kwInNorm.size();i++){
			writer1.append(kwInNorm.get(i)+"\t"+kwTimesInNorm.get(i)+"\n");
			//System.out.println(kwInAnom.get(i)+" "+kwTimesInAnom.get(i));
		}
		for(i=0;i<valueInNorm.size();i++){
			writer2.append(valueInNorm.get(i)+"\t"+valueTimesInNorm.get(i)+"\n");
		}
		writer1.flush();
		writer1.close();
		writer2.flush();
		writer2.close();
	}
	//
	private void getTF(){//TP,TN,FP,FN
		System.out.println(shrshd);
		System.out.println(abTest);
		System.out.println(nmTest);
		System.out.println(abCorr);
		System.out.println(nmCorr);
		TF[0]=(float)abCorr/abTest;
		TF[1]=(float)nmCorr/nmTest;
		TF[2]=(float)(1-TF[1]);
		TF[3]=(float)(1-TF[0]);
		for(int i=0;i<4;i++){
			System.out.println(TF[i]);
		}
	}
	private void detect(float pp, String label){
		if(label.equals("anom")){
			abTest++;
			if(pp>=shrshd){
				abCorr++;
			}
		}
		else{
			nmTest++;
			if(pp<shrshd){
				nmCorr++;
			}
		}
	}
	// bayes for training set
	private void Trainbayes(){
		for(int i = 0;i<abNum.size();i++){
			timesOfAb+=abNum.get(i);
			timesOfNm+=nmNum.get(i);
		}
		//System.out.println(ab);
		//System.out.println(nm);
		//System.out.println(timesOfAb);
		//System.out.println(timesOfNm);
		float[]p = new float[3];
		p[1]=(float)ab/(ab+nm);
		for(int i = 0;i<abNum.size();i++){
			if(abNum.get(i)==0){
				p[0]=(float)0;
			}
			else{p[0]=(float)abNum.get(i)/timesOfAb;}
			p[2]=(float)(abNum.get(i)+nmNum.get(i))/(timesOfAb+timesOfNm);
			float pb = (float) p[0]*p[1]/p[2];
			prob.add(pb);
		}
		System.out.println(prob.size());
		System.out.println(prob);
	}
	private void getGrams(Instances trainInst){
		abTest=0;
		nmTest=0;
		abCorr=0;
		nmCorr=0;
		ab=0;
		nm=0;
		timesOfAb=0;
		timesOfNm=0;
		grams.clear();
		abNum.clear();
		nmNum.clear();
		prob.clear();
		bayesian.clear();
		for(int i=0;i<trainInst.numInstances();i++)
		{
			if(trainInst.instance(i).toString(LABEL).equals("anom")){
				ab++;
			}
			else nm++;
			String temp = trainInst.instance(i).toString(PAL);
			if(temp.charAt(0)=='\''){temp=temp.replaceAll("\'", "");}
			//System.out.println(temp);
			String[]str=temp.split(" ");// each gram
			for(int j=0;j<str.length;j++){
				//System.out.println(str[j]);
				if(str[j].contains("=")){
					String[] substr = str[j].split("=");//substr[0] is gram before = sign
					str[j]=substr[0]+"=";
				};
				//System.out.println(str[j]);
				if(grams.contains(str[j])){
					int idx = grams.indexOf(str[j]);
					if(trainInst.instance(i).toString(LABEL).equals("anom")){
						abNum.set(idx,abNum.get(idx)+1);
					}
					else{
						nmNum.set(idx,nmNum.get(idx)+1);
					}
						
				}
				else{
					grams.add(str[j]);
					if(trainInst.instance(i).toString(LABEL).equals("anom")){
						abNum.add(1);
						nmNum.add(0);
					}
					else{
						abNum.add(0);
						nmNum.add(1);
					}
				}
			}
			System.out.println(i);
		}
		System.out.println(grams);
		System.out.println(abNum);
		System.out.println(nmNum);
		Trainbayes();
	}
	public void getBayes(Instances testInst){
		int idx=0;
		float pp=1;
		for(int i=0;i<testInst.numInstances();i++){
			String temp = testInst.instance(i).toString(PAL);
			temp=temp.replaceAll("\'","");
			String[] str=temp.split(" ");
			float []p = new float[str.length];
			//System.out.println(str.length);
			for(int j=0;j<str.length;j++){
				if(str[j].contains("=")){
					String[] substr=str[j].split("=");
					str[j]=substr[0]+"=";
				}
				if(grams.contains(str[j])){
					idx=grams.indexOf(str[j]);
					p[j]=(float) prob.get(idx);
				}
				else p[j]=1;
				pp=(float)pp*p[j];
				//System.out.println(str[j]);
			}
			bayesian.add(pp);
			String label=testInst.instance(i).toString(LABEL);
			detect(pp,label);
			pp=1;
		}
		System.out.println(bayesian);
		getTF();
	}
	// add one gram to arrayList
	private void addGram(String str, boolean isab){
		if(grams.contains(str)){
			int idx = grams.indexOf(str);
			
			if(isab){
				abNum.set(idx, abNum.get(idx)+1);
			}
			else{
				nmNum.set(idx, nmNum.get(idx)+1);
			}
		}
		else{
			grams.add(str);
			if(isab){
				abNum.add(1);
				nmNum.add(0);
			}
			else{
				nmNum.add(1);
				abNum.add(0);
			}
		}
	}
	// n grams for one instance
	private void Ngrams(int n,String str,String label){
		int l = str.length();
		boolean isab = label.equals("anom");
		if(l<n||l==n){
			addGram(str,isab);
		}
		else{
			int times = l-n+1;
			for(int i=0;i<times;i++){
				String temp="";
				for(int j=0;j<n;j++){
					temp+=str.charAt(i+j);
				}
				addGram(temp,isab);
			}
		}
	}
	private void getNgrams(int n,Instances trainInst,Instances testInst){
		
		abTest=0;
		nmTest=0;
		abCorr=0;
		nmCorr=0;
		ab=0;
		nm=0;
		timesOfAb=0;
		timesOfNm=0;
		grams.clear();
		abNum.clear();
		nmNum.clear();
		prob.clear();
		bayesian.clear();
		
		String str="";
		String label="";
		int l=0;
		for(int i=0;i<trainInst.numInstances();i++){
			label=trainInst.instance(i).toString(LABEL);
			if(label.equals("anom")){
				ab++;
			}
			else nm++;
			str = trainInst.instance(i).toString(PAL);
			//if(str.charAt(0)=='\''){str=str.replaceAll("\'", "");}
			Ngrams(n,str,label);
			System.out.println(i);
		}
		//System.out.println(grams.size());
		//System.out.println(grams);
		//System.out.println(abNum.size());
		//System.out.println(abNum);
		//System.out.println(nmNum.size());
		//System.out.println(nmNum);
		Trainbayes();
		for(int i=0;i<testInst.numInstances();i++){
			float pp=1;
			label=testInst.instance(i).toString(LABEL);
			str = testInst.instance(i).toString(PAL);
			//str=str.replaceAll("\'","");
			l=str.length();
			if(l<n||l==n){
				if(grams.contains(str)){
					pp=(float)(pp*prob.get(grams.indexOf(str)));
					pp=(float)Math.sqrt(pp);
				}
				//else{pp*=1;}
				bayesian.add(pp);
				detect(pp,label);
			}
			else{
				int times=l-n+1;
				for(int j=0;j<times;j++){
					String temp="";
					for(int k=0;k<n;k++){
						temp+=str.charAt(k+j);
					}
					if(grams.contains(temp)){
						pp=(float)(pp*prob.get(grams.indexOf(temp)));// final percentage number.
						pp=(float)Math.sqrt(pp);
					}
					//else{pp*=1;}
				}
				bayesian.add(pp);
				detect(pp,label);
			}
		}
		//System.out.println(bayesian.size());
		//System.out.println(bayesian);
		getTF();
	}
	public void statistic(){
		int n = 0;
		int i=0;
		for(i=0;i<grams.size();i++){
			n+=abNum.get(i)+nmNum.get(i);
		}
		System.out.println("total number of grams:"+n);
		System.out.println("number of distinct grams:"+i);
	}
	public void showPld (Instances trainst, String path) throws IOException{
		FileWriter writer;
		writer = new FileWriter(path + "new.txt"); 
		for(int i=0;i<trainst.numInstances();i++){
			System.out.println(trainst.instance(i).toString(1));
			if(trainst.instance(i).toString(1).equals("?")){
				i++;
			}
			else{
				writer.append(trainst.instance(i).toString()+"\n");
			}
		}
        writer.flush();
        writer.close();
	}
	// new 5 grams
	// Total pacak number, total 
	private ArrayList<String> GramInAnom= new ArrayList();
	private ArrayList<String> GramInNorm= new ArrayList();
	private int[] PackNum=new int[3];//0 is AnomNum, 1 is NormNum;//2 is total
	private ArrayList<Float> kwBayesian=new ArrayList();
	private ArrayList<Float> normPackBayesian=new ArrayList();
	private ArrayList<Float> AnomPackBayesian=new ArrayList();
	private int []tp=new int[4];//tp,tn,fp,fn
	// KwInAnom<>,kwInNorm<>,anNum<>,nmNum<>, Beyesian<>, kwTimesInAnom<>;
	//for testNew5gram();
	private ArrayList<Float> bayesianResult=new ArrayList();
	public void getNew5gram(String value, int n, String isNorm){
		value=value.replaceAll("\'", "");
		value=value.replaceAll(" ","");
		int gramNum=0;
		if(value.length()%n==0){
			gramNum=value.length()/n;
		}
		else{
			gramNum=(value.length()/n)+1;
		}
		int idx=0;
		String []gram= new String[gramNum];
		for(int i=0;i<gramNum;i++){
			gram[i]="";
		}
		for(int i=0;i<gramNum;i++){
			idx=i*n;
			while(idx<(i+1)*n&&idx<value.length()){
				gram[i]+=value.charAt(idx);
				idx++;
			}
			if(isNorm.equals("anom")){
				if(GramInAnom.contains(gram[i])){
					abNum.set(GramInAnom.indexOf(gram[i]), abNum.get(GramInAnom.indexOf(gram[i]))+1);
				}
				else{
					GramInAnom.add(gram[i]);
					abNum.add(1);
				}
			}
			else{
				if(GramInNorm.contains(gram[i])){
					nmNum.set(GramInNorm.indexOf(gram[i]), nmNum.get(GramInNorm.indexOf(gram[i]))+1);
				}
				else{
					GramInNorm.add(gram[i]);
					nmNum.add(1);
				}
			}
		}
		//System.out.println("getGramDone");
	}
	public void kwInNew5gram(String kw, String isNorm){
		if(isNorm.equals("anom")){
			if(kwInAnom.contains(kw)){
				kwTimesInAnom.set(kwInAnom.indexOf(kw), kwTimesInAnom.get(kwInAnom.indexOf(kw))+1);
			}
			else
			{
				kwInAnom.add(kw);
				kwTimesInAnom.add(1);
			}
		}
		else{
			if(kwInNorm.contains(kw)){
				kwTimesInNorm.set(kwInNorm.indexOf(kw), kwTimesInNorm.get(kwInNorm.indexOf(kw))+1);
			}
			else
			{
				kwInNorm.add(kw);
				kwTimesInNorm.add(1);
			}
		}
		//System.out.println("kwDone");
	}
	public void getBayesInNew5gram(){
		float p=0;
		for(int i=0;i<kwInAnom.size();i++){
			if(kwInNorm.contains(kwInAnom.get(i))){
				p= (float) kwTimesInAnom.get(i)/(kwTimesInAnom.get(i)+kwTimesInNorm.get(kwInNorm.indexOf(kwInAnom.get(i))));
			}
			else p=1;
			kwBayesian.add(p);
		}
		for(int i=0;i<GramInAnom.size();i++){
			if(GramInNorm.contains(GramInAnom.get(i))){
				p=(float) abNum.get(i)/(abNum.get(i)+nmNum.get(GramInNorm.indexOf(GramInAnom.get(i))));
			}
			else p=1;
			bayesian.add(p);
		}
		System.out.println("bayesDone");
	}
	public void show(String path) throws IOException{
		FileWriter writer = new FileWriter(path+"show.txt");
		writer.append("kwInAnom:"+"\n");
		for(int i=0;i<kwInAnom.size();i++){
			writer.append(kwInAnom.get(i)+" "+kwTimesInAnom.get(i)+" "+kwBayesian.get(i)+"\n");
		}
		writer.append("kwInNorm:"+"\n");
		for(int i=0;i<kwInNorm.size();i++){
			writer.append(kwInNorm.get(i)+" "+kwTimesInNorm.get(i)+"\n");
		}
		writer.append("gramInAnom:"+"\n");
		for(int i=0;i<GramInAnom.size();i++){
			writer.append(GramInAnom.get(i)+" "+abNum.get(i)+" "+bayesian.get(i)+"\n");
		}
		writer.append("gramInNorm:"+"\n");
		for(int i=0;i<GramInNorm.size();i++){
			writer.append(GramInNorm.get(i)+" "+nmNum.get(i)+"\n");
		}
		writer.flush();
		writer.close();
	}
	public void Train5gram(Instances Inst, int maxN, String path) throws IOException{
		int i=0;
		int idx=0;
		int pNum;
		String pLoad;
		String isNorm;
		String kw;
		String value;
		int  p=0;
		for(i=0;i<Inst.numInstances();i++){
			pNum=Integer.parseInt(Inst.instance(i).toString(0));
			isNorm=Inst.instance(i).toString(2);
			if(isNorm.equals("anom")) PackNum[1]++;
			else PackNum[0]++;
			while(i<Inst.numInstances()&&(idx=Integer.parseInt(Inst.instance(i).toString(0)))==pNum){
				pLoad=Inst.instance(i).toString(1);
				String[]temp=pLoad.split("=");
				kw=temp[0];
				value=temp[1];
				kwInNew5gram(kw,isNorm);
				getNew5gram(value,maxN,isNorm);
				i++;
			}
			p++;
			System.out.println(p);
			i--;
		}
		getBayesInNew5gram();
		show(path);
	}
	public void testNew5gram(Instances Inst, int maxN, String path) throws IOException{
		FileWriter writer = new FileWriter(path+"ResultBynew5gram.txt");
		FileWriter writer2=new FileWriter(path+"ResultBynew5gram1.txt");
		FileWriter writer3=new FileWriter(path+"diagraph-norm.txt");
		FileWriter writer4=new FileWriter(path+"diagraph-anom.txt");
		int i=0;
		int idx=0;
		int pNum;
		String pLoad;
		String isNorm;
		String kw;
		String value;
		float p;
		ArrayList<Float> kwP = new ArrayList();
		for(i=0;i<PackNum.length;i++){
			PackNum[i]=0;
		}
		for(i=0;i<tp.length;i++){
			tp[i]=0;
		}
		for(i=0;i<Inst.numInstances();i++){
			pNum=Integer.parseInt(Inst.instance(i).toString(0));
			isNorm=Inst.instance(i).toString(2);
			PackNum[2]++;
			if(isNorm.equals("anom")) PackNum[1]++;
			else PackNum[0]++;
			writer.append("\n"+"pack"+PackNum[2]+"("+isNorm+")"+":"+"\n");
			float MaxAveP=0;
			float PackAveP=0;
			float aveP = 0;
			while(i<Inst.numInstances()&&(idx=Integer.parseInt(Inst.instance(i).toString(0)))==pNum){
				pLoad=Inst.instance(i).toString(1);
				String[]temp=pLoad.split("=");
				kw=temp[0];
				value=temp[1];
				writer.append("kw"+i+": "+"\n");
				if(kwInAnom.contains(kw)){
					p=kwBayesian.get(kwInAnom.indexOf(kw));
					writer.append(kw+" "+p+"\n");
				}
				else{
					if(kwInNorm.contains(kw)) {
						p=0;
						writer.append(kw+" "+p+"\n");
					}
					else{
						p=1;
						writer.append(kw+" "+p+"(NEW)"+"\n");
					}
					
				}
				value=value.replaceAll("\'", "");
				value=value.replaceAll(" ","");
				int gramNum=0;
				if(value.length()%maxN==0){
					gramNum=value.length()/maxN;
				}
				else{
					gramNum=(value.length()/maxN)+1;
				}
				int idxOfvalue=0;
				String []gram= new String[gramNum];
				for(int j=0;j<gramNum;j++){
					gram[j]="";
				}
				aveP=0;
				PackAveP=0;
				for(int j=0;j<gramNum;j++){
					idxOfvalue=j*maxN;
					while(idxOfvalue<(j+1)*maxN&&idxOfvalue<value.length()){
						gram[j]+=value.charAt(idxOfvalue);
						idxOfvalue++;
					}
					writer.append("Value"+j+": "+"\n");
					if(GramInAnom.contains(gram[j])){
						p=bayesian.get(GramInAnom.indexOf(gram[j]));
						writer.append(gram[j]+" "+p+"\n");
						aveP+=p*p;
					}
					else{
						if(GramInNorm.contains(gram[j])){
							p=0;
							writer.append(gram[j]+" "+p+"\n");
							aveP+=p*p;
						}
						else{
							p=1;
							writer.append(gram[j]+" "+p+"(new)"+"\n");
							aveP+=p*p;
						}
					}
				}
				aveP= (float) Math.sqrt(aveP/gramNum);
				//System.out.println(aveP);
				kwP.add(aveP);
				i++;
			}
			for(int n=0;n<kwP.size();n++){
				if(kwP.get(n)>MaxAveP||kwP.get(n)==MaxAveP){
					MaxAveP=kwP.get(n);
				}
				PackAveP+=kwP.get(n)*kwP.get(n);
			}
			PackAveP=(float)Math.sqrt(PackAveP/kwP.size());
			//if(kwP.size()!=0) PackAveP=(float)PackAveP/kwP.size();
			//if(kwP.size()==0) PackAveP=1;
			writer2.append("pack"+PackNum[2]+"+("+isNorm +"): "+MaxAveP+" "
					+PackAveP+"\n");
			//System.out.println(MaxAveP);
			if(isNorm.equals("norm")) writer3.append(PackAveP+"\n");
			if(isNorm.equals("anom")) writer4.append(PackAveP+"\n");
			if(PackAveP>shrshd||PackAveP==shrshd){// attack predicted
				label_twoClass.add(1);
				if(isNorm.equals("anom")){
					tp[0]++;//tp
				}
				else
					tp[2]++;//fp
			}
			else{// normal predicted
				label_twoClass.add(0);
				if(isNorm.equals("norm")){
					tp[1]++;//tn
				}
				else
					tp[3]++;//fn
			}
			kwP.clear();
			System.out.println(PackNum[2]);
			i--;
		}
		float tpr = (float) tp[0]/(tp[0]+tp[3]);//tp/ tp+fn
		float fpr = (float) tp[2]/(tp[1]+tp[2]);//fp/ fp+tn
		System.out.println(PackNum[0]);
		System.out.println(PackNum[1]);
		System.out.println(tpr);
		System.out.println(fpr);
		System.out.println((float)tp[0]/PackNum[1]);
		System.out.println((float)tp[2]/PackNum[0]);
		writer2.append("tpr:"+tpr+"  fpr:"+fpr+"\n");
		writer.flush();
		writer.close();
		writer2.flush();
		writer2.close();
		writer3.flush();
		writer3.close();
		writer4.flush();
		writer4.close();
	}
	public void combinedResult(){
		System.out.println(label_real.size() +" "+ label_oneClass.size()+ " " + label_twoClass.size());
		int tp=0, fp=0;
		int i=0,j=0;
		int anom_num=0;
		int norm_num=0;
		while(j<label_real.size()){
			if(label_real.get(j)==1){
				anom_num++;
				if(label_oneClass.get(j)==1 || label_twoClass.get(j)==1){
					//combined_result.add(1);
					tp++;
				}
				else {
					//combined_result.add(0);
				}
				j++;
			}
			else{
				norm_num++;
				if(label_oneClass.get(j)==0 || label_twoClass.get(j)==0){
					//combined_result.add(0);
				}
				else {
					//combined_result.add(1);
					fp++;
				}				
				j++;
			}
		}
		double tpr,fpr;
		tpr=(double) tp/anom_num;
		fpr=(double) fp/norm_num;
		i=0;
		j=0;
		int tp1=0,fp1=0;
		double tpr1,fpr1;
		while(j<label_oneClass.size()){
			if(label_oneClass.get(j)==1 || label_twoClass.get(j)==1 ){
				combined_result.add(1);
			}
			else combined_result.add(0);
			if(label_real.get(j)==1){
				if(combined_result.get(j)==1){
					tp1++;
				}
			}
			else{
				if(combined_result.get(j)==1){
					fp1++;
				}
			}
			j++;
		}
		tpr1=(double) tp1/anom_num;
		fpr1=(double) fp1/norm_num;		
		System.out.println("TP: "+ tpr1);
		System.out.println("FP: "+ fpr1);		
	}
	public bayes(String path, String trainFileName, String testFileName, String outFileName )throws Exception{
		try{
			FileReader reader;
            FileWriter writer1;
            FileWriter writer2;
            // read the training instances:
            reader = new FileReader(path + trainFileName);
            Instances trainInst = new Instances(reader);

            // read the test instances:
            reader = new FileReader(path + testFileName);
            Instances testInst = new Instances(reader);
            
            //getGrams(trainInst);//
            //getBayes(testInst);//these two are old gram method. Should be run together.
            
           /* getNgrams(5,trainInst,testInst);//5-grams, 5 can be changed.
            writer1 = new FileWriter(path + outFileName); 
            for(int i=0;i<bayesian.size();i++){
            	writer1.append(bayesian.get(i) + "\n");
            }
            writer1.flush();
            writer1.close();
            writer2 = new FileWriter(path+"result/" + "result.txt");//show grams and show times of each. 
            writer2.append("Threshold:"+"\t"+shrshd+"\n");
            writer2.append("True Positive:"+"\t"+TF[0]+"\n");
            writer2.append("True Negtive:"+"\t"+TF[1]+"\n");
            writer2.append("False Positive:"+"\t"+TF[2]+"\n");
            writer2.append("False Negtive:"+"\t"+TF[3]+"\n");
            for(int i=0;i<bayesian.size();i++){
            	writer2.append(i+":    "+bayesian.get(i)+"\n");
            }
            //for(int i=0;i<grams.size();i++){
            	//writer2.append(grams.get(i)+"\t"+abNum.get(i)+"\t"+nmNum.get(i)+"\n");
            //}
            writer2.flush();
            writer2.close();
            statistic();
			*/
            //showPld(trainInst,path);
            //getAnomKW(trainInst,path);
            //getNormKW(testInst,path);
            //getFirstKwAnom(trainInst,path);
            //getFirstKwNorm(testInst,path);
            //TrainFeature(trainInst);
            //TestFeature(testInst,path);
            
            Train5gram(trainInst,5,path);
            testNew5gram(testInst,5,path);
            TrainFeature(trainInst);
            TestFeature(testInst, path);
            combinedResult();
		}catch (Exception e) {
            System.err.println(e.getMessage());
		}
	}
    public static void main(String[] args) {
    	 
        try  {
                
                String path = "C:/Users/Admin/Desktop/test data/bayes/";// path for the data file
                new bayes( path,  "newTest5050.arff", "NewTest5050t.arff", "bayes1.txt");// Training, Test, Output file contains all grams and show times.
                System.out.println("done");

        } catch(Exception e) {
        }
}
	
	
}

