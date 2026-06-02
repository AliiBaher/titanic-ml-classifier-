import weka.core.*;
import weka.core.converters.CSVLoader;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.*;
import weka.classifiers.functions.*;
import weka.filters.*;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class Main extends JFrame {

    // ── UI fields ──────────────────────────────────────────────
    private final JTextField filePathField = new JTextField(32);
    private final JButton    browseBtn     = new JButton("BROWSE");
    private final JLabel     bestAlgoLabel = new JLabel(" ");
    private final JPanel     inputPanel    = new JPanel();
    private       JLabel     resultLabel;

    // ── WEKA state ─────────────────────────────────────────────
    private Instances        trainSet;      // filtered dataset the best model was trained on
    private Instances        rawStructure;  // pre-filter structure (for building new instances)
    private Classifier       bestModel;
    private Filter           bestFilter;    // Normalize or Discretize — applied before classify
    private final List<JComponent> inputFields = new ArrayList<>();

    // ── Friendly display names for known attributes ────────────
    private static final Map<String, String> ATTR_LABELS    = new HashMap<>();
    private static final Map<String, String> ATTR_TOOLTIPS  = new HashMap<>();
    private static final Map<String, String> CLASS_LABELS   = new HashMap<>();
    static {
        ATTR_LABELS.put("Pclass",   "Passenger Class (1/2/3)");
        ATTR_LABELS.put("Sex",      "Gender  (0=Male, 1=Female)");
        ATTR_LABELS.put("Age",      "Age (years)");
        ATTR_LABELS.put("SibSp",    "Siblings / Spouses Aboard");
        ATTR_LABELS.put("Parch",    "Parents / Children Aboard");
        ATTR_LABELS.put("Fare",     "Ticket Fare (£)  [0 – 512]");
        ATTR_LABELS.put("Embarked", "Port  (0=Southampton, 1=Cherbourg, 2=Queenstown)");

        // tooltips shown on hover
        ATTR_TOOLTIPS.put("Fare",
            "<html>Enter the ticket price in British pounds (£).<br>"
            + "Valid range: <b>0 – 512</b><br>"
            + "Typical fares by class:<br>"
            + "&nbsp;&nbsp;3rd class : £7 – £25<br>"
            + "&nbsp;&nbsp;2nd class : £10 – £65<br>"
            + "&nbsp;&nbsp;1st class : £30 – £512</html>");
        ATTR_TOOLTIPS.put("Age",
            "<html>Passenger age in years.<br>Valid range: <b>0 – 80</b></html>");
        ATTR_TOOLTIPS.put("Pclass",
            "<html>Ticket class:<br>&nbsp;&nbsp;1 = 1st class (upper)<br>&nbsp;&nbsp;2 = 2nd class (middle)<br>&nbsp;&nbsp;3 = 3rd class (lower)</html>");
        ATTR_TOOLTIPS.put("SibSp",
            "<html>Number of siblings or spouses aboard.<br>Valid range: <b>0 – 8</b></html>");
        ATTR_TOOLTIPS.put("Parch",
            "<html>Number of parents or children aboard.<br>Valid range: <b>0 – 6</b></html>");

        // class labels for Titanic
        CLASS_LABELS.put("0", "0 — Did NOT survive");
        CLASS_LABELS.put("1", "1 — Survived");
    }

    // ══════════════════════════════════════════════════════════
    public Main() {
        setTitle("DEUCENG - ML Classification Tool");
        setDefaultCloseOperation(EXIT_ON_CLOSE);

        // ── top bar ──
        JPanel top = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 6));
        top.add(filePathField);
        top.add(browseBtn);

        // ── result label ──
        bestAlgoLabel.setBorder(new EmptyBorder(2, 6, 2, 6));

        // ── center: dynamically filled ──
        inputPanel.setLayout(new BoxLayout(inputPanel, BoxLayout.Y_AXIS));

        // ── assemble ──
        JPanel content = new JPanel();
        content.setLayout(new BoxLayout(content, BoxLayout.Y_AXIS));
        content.add(bestAlgoLabel);
        content.add(inputPanel);

        add(top,     BorderLayout.NORTH);
        add(content, BorderLayout.CENTER);

        browseBtn.addActionListener(e -> browse());

        setMinimumSize(new Dimension(480, 110));
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    // ── File chooser ──────────────────────────────────────────
    private void browse() {
        JFileChooser fc = new JFileChooser(".");
        if (fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File f = fc.getSelectedFile();
            filePathField.setText(f.getAbsolutePath());
            runClassification(f);
        }
    }

    // ══════════════════════════════════════════════════════════
    // Background worker: load data → build 10 models → find best
    // ══════════════════════════════════════════════════════════
    private void runClassification(File f) {
        browseBtn.setEnabled(false);
        bestAlgoLabel.setText("Processing… please wait.");
        inputPanel.removeAll();
        revalidate(); repaint();

        new SwingWorker<Void, Void>() {

            double     localBest   = -1;
            Classifier localModel;
            String     localName;
            Instances  localTrain;
            Filter     localFilter;
            Instances  localRaw;
            String     err;

            @Override
            protected Void doInBackground() {
                try {
                    // ── 1. Load CSV ──────────────────────────
                    System.out.println("[1] Loading CSV: " + f.getAbsolutePath());
                    CSVLoader loader = new CSVLoader();
                    loader.setSource(f);
                    Instances raw = loader.getDataSet();
                    raw.setClassIndex(0);
                    System.out.println("[1] Loaded: " + raw.numInstances() + " instances, " + raw.numAttributes() + " attributes");

                    // ── 2. Class: numeric → nominal ──────────
                    System.out.println("[2] NumericToNominal on class...");
                    NumericToNominal n2n = new NumericToNominal();
                    n2n.setAttributeIndices("first");
                    n2n.setInputFormat(raw);
                    Instances fixed = Filter.useFilter(raw, n2n);
                    fixed.setClassIndex(0);
                    System.out.println("[2] Done. Class type: " + (fixed.classAttribute().isNominal() ? "nominal" : "numeric"));

                    // ── 3. Nominal dataset (Discretize) ──────
                    System.out.println("[3] Discretize...");
                    Discretize disc = new Discretize();
                    disc.setAttributeIndices("2-last");  // skip class at index 0
                    disc.setInputFormat(fixed);
                    Instances nomData = Filter.useFilter(fixed, disc);
                    nomData.setClassIndex(0);
                    System.out.println("[3] Done.");

                    // ── 4. Numeric + Normalize ────────────────
                    System.out.println("[4] Normalize numeric data...");
                    Normalize norm = new Normalize();
                    norm.setInputFormat(fixed);
                    Instances numNorm = Filter.useFilter(fixed, norm);
                    numNorm.setClassIndex(0);
                    System.out.println("[4] Done.");

                    // keep a 0-instance copy of 'fixed' for building new instances
                    localRaw = new Instances(fixed, 0);

                    // ── 5. Ten models ─────────────────────────
                    IBk knn1 = new IBk(); knn1.setKNN(1);
                    IBk knn3 = new IBk(); knn3.setKNN(3);
                    IBk knn5 = new IBk(); knn5.setKNN(5);

                    Classifier[] models = {
                        new NaiveBayes(),
                        new J48(),
                        new RandomForest(),
                        new RandomTree(),
                        knn1, knn3, knn5,
                        new SMO(),
                        new Logistic(),
                        new MultilayerPerceptron()
                    };
                    String[] names = {
                        "Naive Bayes",
                        "J48",
                        "Random Forest",
                        "Random Tree",
                        "KNN K=1", "KNN K=3", "KNN K=5",
                        "SVM (SMO)",
                        "Logistic Regression",
                        "Neural Network (MLP)"
                    };
                    boolean[] useNum = {
                        false, false, false, false,
                        true, true, true, true, true, true
                    };

                    // ── 6. Train + 10-fold CV ─────────────────
                    for (int i = 0; i < models.length; i++) {
                        Instances train = useNum[i] ? numNorm : nomData;
                        models[i].buildClassifier(train);
                        Evaluation eval = new Evaluation(train);
                        eval.crossValidateModel(models[i], train, 10, new Random(1));
                        double acc = eval.pctCorrect();
                        System.out.printf("Approach #%d: %s -> Correct: %d / Accuracy: %.2f%%%n",
                                i + 1, names[i], (int) eval.correct(), acc);
                        if (acc > localBest) {
                            localBest   = acc;
                            localModel  = models[i];
                            localName   = names[i];
                            localTrain  = train;
                            localFilter = useNum[i] ? norm : disc;
                        }
                    }

                } catch (Throwable ex) {
                    err = ex.toString();
                    ex.printStackTrace();
                }
                return null;
            }

            @Override
            protected void done() {
                browseBtn.setEnabled(true);
                if (err != null) {
                    bestAlgoLabel.setText("Error — see dialog");
                    JOptionPane.showMessageDialog(Main.this, err, "Classification Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if (localName == null) {
                    JOptionPane.showMessageDialog(Main.this, "No model was trained. Check terminal for stack trace.", "No Model", JOptionPane.WARNING_MESSAGE);
                    bestAlgoLabel.setText("Error: no model trained.");
                    return;
                }
                bestModel    = localModel;
                trainSet     = localTrain;
                rawStructure = localRaw;
                bestFilter   = localFilter;
                bestAlgoLabel.setText(String.format(
                        "%s is the most successful algorithm for this data set (%%%s)",
                        localName, String.format("%.2f", localBest)));
                buildInputPanel();
            }
        }.execute();
    }

    // ══════════════════════════════════════════════════════════
    // Dynamically build input fields from the best model's dataset
    // Nominal attribute  → JComboBox  (dropdown with bin values)
    // Numeric attribute  → JTextField
    // ══════════════════════════════════════════════════════════
    private void buildInputPanel() {
        inputPanel.removeAll();
        inputFields.clear();

        JPanel grid = new JPanel(new GridBagLayout());
        grid.setAlignmentX(LEFT_ALIGNMENT);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets  = new Insets(3, 8, 3, 8);
        gbc.fill    = GridBagConstraints.HORIZONTAL;
        gbc.anchor  = GridBagConstraints.WEST;

        for (int i = 1; i < trainSet.numAttributes(); i++) {   // skip class at 0
            Attribute attr = trainSet.attribute(i);
            String label = ATTR_LABELS.getOrDefault(attr.name(), attr.name());

            gbc.gridx = 0; gbc.gridy = i - 1; gbc.weightx = 0;
            grid.add(new JLabel(label), gbc);

            JComponent field;
            if (attr.isNominal()) {
                JComboBox<String> cb = new JComboBox<>();
                for (int j = 0; j < attr.numValues(); j++) cb.addItem(attr.value(j));
                field = cb;
            } else {
                field = new JTextField(10);
            }

            String tip = ATTR_TOOLTIPS.get(attr.name());
            if (tip != null) field.setToolTipText(tip);

            gbc.gridx = 1; gbc.weightx = 1;
            grid.add(field, gbc);
            inputFields.add(field);
        }

        inputPanel.add(grid);

        // ── Discover row ──
        JButton discoverBtn = new JButton("Discover");
        resultLabel = new JLabel("RESULT : ");

        JPanel bottom = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 6));
        bottom.setAlignmentX(LEFT_ALIGNMENT);
        bottom.add(discoverBtn);
        bottom.add(resultLabel);
        inputPanel.add(bottom);

        discoverBtn.addActionListener(e -> discover());

        pack();
        revalidate();
    }

    // ══════════════════════════════════════════════════════════
    // Classify the instance built from the current input values
    // ══════════════════════════════════════════════════════════
    private void discover() {
        try {
            // Build instance from raw (pre-filter) structure
            double[] vals = new double[rawStructure.numAttributes()];
            vals[0] = Utils.missingValue();   // class is unknown

            for (int i = 1; i < rawStructure.numAttributes(); i++) {
                Attribute  attr = rawStructure.attribute(i);
                JComponent comp = inputFields.get(i - 1);

                if (attr.isNominal()) {
                    String sel = ((JComboBox<?>) comp).getSelectedItem().toString();
                    vals[i] = attr.indexOfValue(sel);
                } else {
                    String txt = ((JTextField) comp).getText().trim();
                    vals[i] = Double.parseDouble(txt);
                }
            }

            // Wrap in a 1-row dataset, apply the same filter used during training
            Instances singleRow = new Instances(rawStructure, 0);
            singleRow.add(new DenseInstance(1.0, vals));
            singleRow.setClassIndex(0);
            Instances filtered = Filter.useFilter(singleRow, bestFilter);
            filtered.setClassIndex(0);
            Instance inst = filtered.instance(0);

            double  cls      = bestModel.classifyInstance(inst);
            String  rawLabel = trainSet.classAttribute().value((int) cls);
            String  friendly = CLASS_LABELS.getOrDefault(rawLabel, rawLabel);
            resultLabel.setText("RESULT : " + friendly);

        } catch (NumberFormatException nfe) {
            resultLabel.setText("Error: please enter valid numbers.");
        } catch (Exception ex) {
            resultLabel.setText("Error: " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    // ══════════════════════════════════════════════════════════
    public static void main(String[] args) {
        SwingUtilities.invokeLater(Main::new);
    }
}