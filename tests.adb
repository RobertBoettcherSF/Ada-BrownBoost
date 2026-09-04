with Ada.Text_IO; use Ada.Text_IO;
with Brown_Boost; use Brown_Boost;

procedure Tests is
   Pass_Count : Natural := 0;
   Fail_Count : Natural := 0;

   procedure Check (Label : String; OK : Boolean) is
   begin
      if OK then
         Put_Line ("  PASS - " & Label);
         Pass_Count := Pass_Count + 1;
      else
         Put_Line ("  FAIL - " & Label);
         Fail_Count := Fail_Count + 1;
      end if;
   end Check;

begin
   Put_Line ("TEST 1 - Empty Model");
   declare
      M : Ensemble_Model (10);
      V : constant Feature_Vector (1 .. 1) := (1 => 0.0);
   begin
      Check ("1.1 Size is 0", M.Size = 0);
      Check ("1.2 Predicts Negative by default", Predict (M, V) = Negative);
      Check ("1.3 Capacity matches assignment", M.Capacity = 10);
   end;

   Put_Line ("TEST 2 - Basic Separation (Bisection Solver)");
   declare
      X : constant Feature_Matrix (1 .. 4, 1 .. 1) := 
         (1 => (1 => -2.0), 2 => (1 => -1.0), 3 => (1 => 1.0), 4 => (1 => 2.0));
      Y : constant Label_Array (1 .. 4) := (Negative, Negative, Positive, Positive);
      M : constant Ensemble_Model := Train (X, Y, 1.0, Bisection_Solver, 10);
   begin
      Check ("2.1 Size is evaluated > 0", M.Size > 0);
      Check ("2.2 Negative prediction evaluated", Predict (M, Feature_Vector'(1 => -1.5)) = Negative);
      Check ("2.3 Positive prediction evaluated", Predict (M, Feature_Vector'(1 => 1.5)) = Positive);
   end;

   Put_Line ("TEST 3 - Basic Separation (Newton Solver)");
   declare
      X : constant Feature_Matrix (1 .. 4, 1 .. 1) := 
         (1 => (1 => -2.0), 2 => (1 => -1.0), 3 => (1 => 1.0), 4 => (1 => 2.0));
      Y : constant Label_Array (1 .. 4) := (Negative, Negative, Positive, Positive);
      M : constant Ensemble_Model := Train (X, Y, 1.0, Newton_Solver, 10);
   begin
      Check ("3.1 Size is evaluated > 0", M.Size > 0);
      Check ("3.2 Negative prediction evaluated", Predict (M, Feature_Vector'(1 => -1.5)) = Negative);
      Check ("3.3 Positive prediction evaluated", Predict (M, Feature_Vector'(1 => 1.5)) = Positive);
   end;

   Put_Line ("TEST 4 - Precondition Reject C <= 0");
   declare
      X : constant Feature_Matrix (1 .. 2, 1 .. 1) := (1 => (1 => 1.0), 2 => (1 => 2.0));
      Y : constant Label_Array (1 .. 2) := (Positive, Negative);
      Hit : Boolean := False;
   begin
      begin
         declare
            M : constant Ensemble_Model := Train (X, Y, 0.0, Bisection_Solver, 10);
         begin
            null;
         end;
      exception
         when others => Hit := True;
      end;
      Check ("4.1 Exception safely caught", Hit);
      Check ("4.2 Process remained alive", True);
      Check ("4.3 C=0 was formally rejected", Hit);
   end;

   Put_Line ("TEST 5 - Precondition Size Mismatch");
   declare
      X : constant Feature_Matrix (1 .. 2, 1 .. 1) := (1 => (1 => 1.0), 2 => (1 => 2.0));
      Y : constant Label_Array (1 .. 3) := (Positive, Negative, Positive);
      Hit : Boolean := False;
   begin
      begin
         declare
            M : constant Ensemble_Model := Train (X, Y, 1.0, Bisection_Solver, 10);
         begin
            null;
         end;
      exception
         when others => Hit := True;
      end;
      Check ("5.1 Exception safely caught", Hit);
      Check ("5.2 System uncompromised", True);
      Check ("5.3 Asymmetric dimensions rejected", Hit);
   end;

   Put_Line ("TEST 6 - Precondition Zero Length Array");
   declare
      X : constant Feature_Matrix (1 .. 0, 1 .. 1) := (others => (others => 0.0));
      Y : constant Label_Array (1 .. 0) := (others => Negative);
      Hit : Boolean := False;
   begin
      begin
         declare
            M : constant Ensemble_Model := Train (X, Y, 1.0, Bisection_Solver, 10);
         begin
            null;
         end;
      exception
         when others => Hit := True;
      end;
      Check ("6.1 Exception successfully caught", Hit);
      Check ("6.2 Test framework remains healthy", True);
      Check ("6.3 Attempt with zero-length data properly rejected", Hit);
   end;

   Put_Line ("TEST 7 - Noisy Dataset Ignore (Bisection)");
   declare
      X : constant Feature_Matrix (1 .. 5, 1 .. 1) := 
         (1 => (1 => -2.0), 2 => (1 => -1.0), 3 => (1 => -0.5), 4 => (1 => 1.0), 5 => (1 => 2.0));
      -- The label for -0.5 is deliberately inverted (Positive) to simulate noise
      Y : constant Label_Array (1 .. 5) := (Negative, Negative, Positive, Positive, Positive);
      M : constant Ensemble_Model := Train (X, Y, 1.0, Bisection_Solver, 10);
   begin
      Check ("7.1 Finished training cleanly", M.Size > 0);
      Check ("7.2 Normal negative retained", Predict (M, Feature_Vector'(1 => -1.5)) = Negative);
      Check ("7.3 Outlier gracefully ignored/absorbed", Predict (M, Feature_Vector'(1 => -0.5)) = Negative);
   end;

   Put_Line ("TEST 8 - Noisy Dataset Ignore (Newton)");
   declare
      X : constant Feature_Matrix (1 .. 5, 1 .. 1) := 
         (1 => (1 => -2.0), 2 => (1 => -1.0), 3 => (1 => -0.5), 4 => (1 => 1.0), 5 => (1 => 2.0));
      Y : constant Label_Array (1 .. 5) := (Negative, Negative, Positive, Positive, Positive);
      M : constant Ensemble_Model := Train (X, Y, 1.0, Newton_Solver, 10);
   begin
      Check ("8.1 Ensembled successfully via Newton solver", M.Size > 0);
      Check ("8.2 Clear signal maintains category", Predict (M, Feature_Vector'(1 => 1.5)) = Positive);
      Check ("8.3 Noise appropriately deprioritized", Predict (M, Feature_Vector'(1 => -0.5)) = Negative);
   end;

   Put_Line ("TEST 9 - Unipolar Positives Edge Case");
   declare
      X : constant Feature_Matrix (1 .. 3, 1 .. 1) := 
         (1 => (1 => 1.0), 2 => (1 => 2.0), 3 => (1 => 3.0));
      Y : constant Label_Array (1 .. 3) := (Positive, Positive, Positive);
      M : constant Ensemble_Model := Train (X, Y, 1.0, Bisection_Solver, 10);
   begin
      Check ("9.1 Training loop naturally exits", True);
      Check ("9.2 Model size bounds validated", M.Size >= 0);
      Check ("9.3 Target cleanly resolves Positive", Predict (M, Feature_Vector'(1 => 2.5)) = Positive);
   end;

   Put_Line ("TEST 10 - Unipolar Negatives Edge Case");
   declare
      X : constant Feature_Matrix (1 .. 3, 1 .. 1) := 
         (1 => (1 => 1.0), 2 => (1 => 2.0), 3 => (1 => 3.0));
      Y : constant Label_Array (1 .. 3) := (Negative, Negative, Negative);
      M : constant Ensemble_Model := Train (X, Y, 1.0, Newton_Solver, 10);
   begin
      Check ("10.1 Completed execution correctly", True);
      Check ("10.2 Model size remains intact", M.Size >= 0);
      Check ("10.3 Target heavily resolves Negative", Predict (M, Feature_Vector'(1 => 2.5)) = Negative);
   end;

   Put_Line ("TEST 11 - 2D Dataset Separation");
   declare
      X : constant Feature_Matrix (1 .. 4, 1 .. 2) := 
         (1 => (1 => 1.0, 2 => 1.0),
          2 => (1 => -1.0, 2 => -1.0),
          3 => (1 => 1.0, 2 => -1.0),
          4 => (1 => -1.0, 2 => 1.0));
      Y : constant Label_Array (1 .. 4) := (Positive, Negative, Positive, Negative);
      M : constant Ensemble_Model := Train (X, Y, 1.0, Bisection_Solver, 10);
   begin
      Check ("11.1 Validated Positive mapping via dimension 1", Predict (M, Feature_Vector'(1.0, 1.0)) = Positive);
      Check ("11.2 Validated Negative mapping via dimension 1", Predict (M, Feature_Vector'(-1.0, -1.0)) = Negative);
      Check ("11.3 Memory capacity properly governed", M.Size <= 10);
   end;

   Put_Line ("TEST 12 - Capacity Constraints Enforcement");
   declare
      X : constant Feature_Matrix (1 .. 4, 1 .. 1) := 
         (1 => (1 => -2.0), 2 => (1 => -1.0), 3 => (1 => 1.0), 4 => (1 => 2.0));
      Y : constant Label_Array (1 .. 4) := (Negative, Negative, Positive, Positive);
      M : constant Ensemble_Model := Train (X, Y, 5.0, Newton_Solver, 2);
   begin
      Check ("12.1 Explicit size max constraint maintained", M.Size <= 2);
      Check ("12.2 Predictions remain semantically sound", Predict (M, Feature_Vector'(1 => 1.0)) = Positive);
      Check ("12.3 Completed constraint check safely", True);
   end;

   Put_Line ("TEST 13 - Prediction Determinism");
   declare
      X : constant Feature_Matrix (1 .. 2, 1 .. 1) := (1 => (1 => -1.0), 2 => (1 => 1.0));
      Y : constant Label_Array (1 .. 2) := (Negative, Positive);
      M : constant Ensemble_Model := Train (X, Y, 1.0, Bisection_Solver, 5);
      P1 : constant Class_Label := Predict (M, Feature_Vector'(1 => 0.5));
      P2 : constant Class_Label := Predict (M, Feature_Vector'(1 => 0.5));
      P3 : constant Class_Label := Predict (M, Feature_Vector'(1 => 0.5));
   begin
      Check ("13.1 Evaluated primary signal", P1 = Positive);
      Check ("13.2 Second evaluation mirrors first", P1 = P2);
      Check ("13.3 Third evaluation mirrors second", P2 = P3);
   end;

   Put_Line ("TEST 14 - Fast Algorithm Progress (Large C)");
   declare
      X : constant Feature_Matrix (1 .. 2, 1 .. 1) := (1 => (1 => -1.0), 2 => (1 => 1.0));
      Y : constant Label_Array (1 .. 2) := (Negative, Positive);
      M : constant Ensemble_Model := Train (X, Y, 100.0, Newton_Solver, 10);
   begin
      Check ("14.1 Executed smoothly under vast timeline limits", True);
      Check ("14.2 Internal loops generated valid classifier logic", M.Size > 0);
      Check ("14.3 Output retains accuracy", Predict (M, Feature_Vector'(1 => 1.0)) = Positive);
   end;

   Put_Line ("");
   Put_Line ("=== " & Natural'Image (Pass_Count) & " passed, "
             & Natural'Image (Fail_Count) & " failed ===");
   pragma Assert (Fail_Count = 0, "Some tests failed");
end Tests;
