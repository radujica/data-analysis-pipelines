import com.google.common.base.Joiner;
import com.google.common.primitives.Floats;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import ucar.nc2.Attribute;
import ucar.nc2.NetcdfFile;
import ucar.nc2.Variable;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

// TODO: convert for loops to streams? no native support for floats though
// https://stackoverflow.com/questions/26951431/how-to-get-an-observablefloatarray-from-a-stream/26970398#26970398
public class Pipeline  {
    private static final String CALENDAR = "proleptic_gregorian";
    private static final String UNITS = "days since 1950-01-01";

    private Object readVariable(Variable variable) throws IOException {
        Class<?> dataType = variable.getDataType().getPrimitiveClassType();
        Object data;

        Attribute scaleFactorAttribute = variable.findAttribute("scale_factor");
        if (scaleFactorAttribute != null) {
            // TODO: delay the conversion to later?
            float scaleFactor = scaleFactorAttribute.getNumericValue().floatValue();
            float[] dataRaw = (float[]) variable.read().get1DJavaArray(float.class);
            for (int i = 0; i < dataRaw.length; i++) {
                dataRaw[i] *= scaleFactor;
            }
            data = dataRaw;
        } else {
            data = variable.read().get1DJavaArray(dataType);
            // time is handled manually later, so keep as int[]
        }

        return data;
    }

    private DataFrame readData(String path) throws IOException {
        NetcdfFile file = NetcdfFile.open(path);

        DataFrame df = new DataFrame();
        for (Variable variable : file.getVariables()) {
            df.add(variable.getShortName(), readVariable(variable));
        }

        return df;
    }

    private void print(DataFrame df) {
        DataFrame printData = new DataFrame(df.getSize());

        for (String column : df.keys()) {
            Object data = df.get(column);

            if (data.getClass().equals(float[].class)) {
                float[] dataRaw = (float[]) data;
                printData.put(column, Joiner.on(", ").join(Floats.asList(dataRaw)));
            } else if (data.getClass().equals(int[].class)) {
                int[] dataRaw = (int[]) data;
                String[] dates = new String[df.getSize()];
                for (int i = 0; i < df.getSize(); i++) {
                    dates[i] = DataFrame.intTimeToString(dataRaw[i], Pipeline.CALENDAR, Pipeline.UNITS);
                }
                printData.put(column, Joiner.on(", ").join(dates));
            }
        }

        Joiner.MapJoiner mapJoiner = Joiner.on("\n").withKeyValueSeparator("=");

        System.out.println(mapJoiner.join(printData.values()));
        System.out.println("");
    }

    private Object computeAbsMaxmin(Object max, Object min) {
        float[] maxData = (float[]) max;
        float[] minData = (float[]) min;

        float[] result = new float[maxData.length];

        for (int i = 0; i < maxData.length; i++) {
            result[i] = Math.abs(maxData[i] - minData[i]);
        }

        return result;
    }

    private Object computeYearMonth(Object time) {
        int[] timeRaw = (int[]) time;
        String[] yearMonth = new String[timeRaw.length];
        for (int i = 0; i < timeRaw.length; i++) {
            yearMonth[i] = DataFrame.intTimeToString(timeRaw[i], CALENDAR, UNITS)
                    .substring(0, 7)
                    .replaceFirst("-", "");
        }

        return yearMonth;
    }

    private void toCsv(DataFrame df, String path) throws IOException {
        BufferedWriter writer = Files.newBufferedWriter(Paths.get(path));
        int numberOfColumns = df.keys().size();
        String[] header = df.keys().toArray(new String[numberOfColumns]);
        CSVPrinter csvPrinter = new CSVPrinter(writer, CSVFormat.DEFAULT.withHeader(header));

        for (int i = 0; i < df.getSize(); i++) {
            Object[] rowData = new Object[numberOfColumns];
            for (int j = 0; j < numberOfColumns; j++) {
                Object columnData = df.get(header[j]);
                if (columnData.getClass().equals(float[].class)) {
                    float[] rawData = (float[]) columnData;
                    rowData[j] = rawData[i];
                } else if (columnData.getClass().equals(int[].class)) {
                    int[] rawData = (int[]) columnData;
                    rowData[j] = DataFrame.intTimeToString(rawData[i], Pipeline.CALENDAR, Pipeline.UNITS);
                } else if (columnData.getClass().equals(String[].class)) {
                    String[] rawData = (String[]) columnData;
                    rowData[j] = rawData[i];
                }
            }
            csvPrinter.printRecord(rowData);
        }

        csvPrinter.close(); // automatically flushes first
    }

    public void start(String input, String slice, String output) throws IOException {
        DataFrame df1 = readData(input + "data1.nc");
        DataFrame df2 = readData(input + "data2.nc");

        // PIPELINE
        // 1. join the 2 dataframes
        DataFrame df = df1.join(df2);

        // 2. quick preview on the data
        DataFrame df_head = df.subset(0, 10);
        if (output == null) {
            print(df_head);
        } else {
            toCsv(df_head, output + "head.csv");
        }

        // 3. subset the data
        String[] slice_ = slice.split(":");
        df = df.subset(Integer.parseInt(slice_[0]), Integer.parseInt(slice_[1]));

        // 4. drop rows with null values
        df = df.filter();

        // 5. drop columns
        df.pop("pp_stderr");
        df.pop("rr_stderr");

        // 6. UDF 1: compute absolute difference between max and min
        df.put("abs_diff", computeAbsMaxmin(df.get("tx"), df.get("tn")));

        // 7. explore the data through aggregations
        DataFrame df_agg = df.aggregations();
        if (output == null) {
            print(df_agg);
        } else {
            toCsv(df_agg, output + "agg.csv");
        }


        // 8. compute mean per month
        // UDF 2: compute custom year+month format
        df.put("year_month", computeYearMonth(df.get("time")));
        // also handles the join back
        df = df.groupBy();
        df.pop("year_month");

        // careful with this! prints all
        if (output == null) {
            print(df);
        } else {
            toCsv(df, output + "result.csv");
        }
    }
}
