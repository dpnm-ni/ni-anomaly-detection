package model.sla_binary;

import java.io.*;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.prediction.*;
import hex.genmodel.MojoModel;

public class sla_binary {
	public static void main(String[] args) throws Exception {
		String modelfile = "XGBoost_binary.zip";
		// String modelfile = "XGBoost_1_AutoML_20190703_181005.zip";
		EasyPredictModelWrapper model = new EasyPredictModelWrapper(MojoModel.load(modelfile));

		RowData row = new RowData();
		row.put("firewall_cpu", args[0]);
		row.put("firewall_mem", args[1]);
		row.put("firewall_disk_read", args[2]);
		row.put("firewall_disk_write", args[3]);
		row.put("firewall_rx_bytes", args[4]);
		row.put("firewall_tx_bytes", args[5]);

		row.put("flowmonitor_cpu", args[6]);
		row.put("flowmonitor_mem", args[7]);
		row.put("flowmonitor_disk_read", args[8]);
		row.put("flowmonitor_disk_write", args[9]);
		row.put("flowmonitor_rx_bytes", args[10]);
		row.put("flowmonitor_tx_bytes", args[11]);

		row.put("dpi_cpu", args[12]);
		row.put("dpi_mem", args[13]);
		row.put("dpi_disk_read", args[14]);
		row.put("dpi_disk_write", args[15]);
		row.put("dpi_rx_bytes", args[16]);
		row.put("dpi_tx_bytes", args[17]);

		row.put("ids_cpu", args[18]);
		row.put("ids_mem", args[19]);
		row.put("ids_disk_read", args[20]);
		row.put("ids_disk_write", args[21]);
		row.put("ids_rx_bytes", args[22]);
		row.put("ids_tx_bytes", args[23]);

		row.put("lb_cpu", args[24]);
		row.put("lb_mem", args[25]);
		row.put("lb_disk_read", args[26]);
		row.put("lb_disk_write", args[27]);
		row.put("lb_rx_bytes", args[28]);
		row.put("lb_tx_bytes", args[29]);

		System.out.println("Row: " + row);

		BinomialModelPrediction p = model.predictBinomial(row);
		System.out.println("SLA violation results: " + p.label);
		System.out.print("Class probabilities: ");
		for (int i = 0; i < p.classProbabilities.length; i++) {
			if (i > 0) {
				System.out.print(",");
			}
			System.out.print(p.classProbabilities[i]);
		}
		System.out.println("");
	}
}
