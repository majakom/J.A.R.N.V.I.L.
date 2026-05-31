import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import LoginPage from "./pages/LoginPage";
import Dashboard from "./pages/Dashboard";
import Instructions from "./pages/ShowInstructions";
import Steps from "./pages/ShowStep";
import ElementDatabase from "./pages/ElementDatabasePage";
import InstructionDatabase from "./pages/InstructionDatabasePage";
import SelectDatabase from "./pages/SelectDatabasePage";
import ShowStep from "./pages/ShowStep";
import EditInstruction from "./pages/EditInstructionPage";
import CreateInstruction from "./pages/CreateInstructionPage";

const PrivateRoute = ({ children }: { children: JSX.Element }) => {
  const token = localStorage.getItem("token");
  return token ? children : <Navigate to="/" />;
};

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LoginPage />} />
        <Route
          path="/dashboard"
          element={
            <PrivateRoute>
              <Dashboard />
            </PrivateRoute>
          }
        />
        <Route
          path="/instructions"
          element={
            <PrivateRoute>
              <Instructions />
            </PrivateRoute>
          }
        />
        <Route
          path="/steps"
          element={
            <PrivateRoute>
              <Steps />
            </PrivateRoute>
          }
        />
        <Route
          path="/elements_database"
          element={
            <PrivateRoute>
              <ElementDatabase />
            </PrivateRoute>
          }
        />
        <Route
          path="/instructions_database"
          element={
            <PrivateRoute>
              <InstructionDatabase />
            </PrivateRoute>
          }
        />
        <Route
          path="/select_database"
          element={
            <PrivateRoute>
              <SelectDatabase />
            </PrivateRoute>
          }
        />
        <Route
          path="/show_instruction/:id"
          element={
            <PrivateRoute>
              <ShowStep />
            </PrivateRoute>
          }
        />
        <Route
          path="/edit_instruction/:id"
          element={
            <PrivateRoute>
              <EditInstruction />
            </PrivateRoute>
          }
        />
        <Route
          path="/create_instruction"
          element={
            <PrivateRoute>
              <CreateInstruction />
            </PrivateRoute>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}